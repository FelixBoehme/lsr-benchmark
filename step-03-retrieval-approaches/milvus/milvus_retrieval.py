#!/usr/bin/env python3
import gzip
import math
import os
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from pymilvus import DataType, MilvusClient
from pymilvus.client.types import LoadState
from pymilvus.exceptions import MilvusException
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings


COLLECTION_NAME = "lsr_benchmark"
SPARSE_VECTOR_FIELD = "embedding"
DOCUMENT_ID_FIELD = "doc_id"
MILVUS_BINARY = "/milvus/bin/milvus"
MILVUS_URI = "http://127.0.0.1:19530"
MAX_SPARSE_INDEX = (1 << 32) - 1
MAX_DOCUMENT_ID_BYTES = 65535


@dataclass(frozen=True)
class MilvusIndex:
    collection_name: str
    document_count: int
    indexed_document_count: int
    algorithm: str


def to_sparse_vector(tokens, values):
    if len(tokens) != len(values):
        raise ValueError("Embedding tokens and values must have the same length.")

    merged = defaultdict(float)
    for token, value in zip(tokens, values):
        try:
            index = int(token)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Sparse vector index {token!r} is not an integer.") from error
        if not 0 <= index <= MAX_SPARSE_INDEX:
            raise ValueError("Sparse vector indices must fit in an unsigned 32-bit integer.")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("Embedding values must be finite.")
        merged[index] += numeric_value

    return {index: value for index, value in sorted(merged.items()) if value != 0}


def batched(items, batch_size):
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def wait_for_index(client, collection_name, index_name, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        description = client.describe_index(collection_name, index_name)
        state = description.get("state")
        if state == "Finished":
            return description
        if state == "Failed":
            raise RuntimeError(f"Milvus index {index_name!r} failed to build.")
        time.sleep(0.1)
    raise TimeoutError(f"Milvus index {index_name!r} was not ready within {timeout} seconds.")


def wait_for_load(client, collection_name, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get_load_state(collection_name).get("state")
        if state == LoadState.Loaded:
            return
        if state == LoadState.NotLoad:
            raise RuntimeError(f"Milvus collection {collection_name!r} failed to load.")
        time.sleep(0.1)
    raise TimeoutError(f"Milvus collection {collection_name!r} was not loaded within {timeout} seconds.")


def build_index(
    client,
    document_embeddings,
    batch_size,
    algorithm,
    drop_ratio_build,
    collection_name=COLLECTION_NAME,
):
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if algorithm not in {"DAAT_MAXSCORE", "DAAT_WAND", "TAAT_NAIVE"}:
        raise ValueError(f"Unsupported Milvus sparse retrieval algorithm: {algorithm}.")
    if not 0 <= drop_ratio_build < 1:
        raise ValueError("Build drop ratio must be at least 0 and less than 1.")
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name=DOCUMENT_ID_FIELD,
        datatype=DataType.VARCHAR,
        max_length=MAX_DOCUMENT_ID_BYTES,
    )
    schema.add_field(
        field_name=SPARSE_VECTOR_FIELD,
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )
    client.create_collection(collection_name, schema=schema)

    seen_document_ids = set()
    pending_documents = []
    document_count = 0
    indexed_document_count = 0

    for internal_id, (document_id, tokens, values) in enumerate(document_embeddings):
        document_id = str(document_id)
        if document_id in seen_document_ids:
            raise ValueError(f"Document IDs must be unique; found duplicate {document_id!r}.")
        if len(document_id.encode("utf-8")) > MAX_DOCUMENT_ID_BYTES:
            raise ValueError(
                f"Document ID {document_id!r} exceeds Milvus's {MAX_DOCUMENT_ID_BYTES}-byte limit."
            )
        seen_document_ids.add(document_id)
        document_count += 1

        vector = to_sparse_vector(tokens, values)
        if not vector:
            continue
        pending_documents.append(
            {
                "id": internal_id,
                DOCUMENT_ID_FIELD: document_id,
                SPARSE_VECTOR_FIELD: vector,
            }
        )
        indexed_document_count += 1
        if len(pending_documents) == batch_size:
            client.insert(collection_name, pending_documents)
            pending_documents = []

    if pending_documents:
        client.insert(collection_name, pending_documents)
    client.flush(collection_name)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=SPARSE_VECTOR_FIELD,
        index_name=SPARSE_VECTOR_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={
            "inverted_index_algo": algorithm,
            "drop_ratio_build": drop_ratio_build,
        },
    )
    client.create_index(collection_name, index_params)
    description = wait_for_index(client, collection_name, SPARSE_VECTOR_FIELD)
    if description.get("metric_type") != "IP":
        raise RuntimeError("Milvus created the sparse index with a non-IP metric.")
    if description.get("inverted_index_algo") != algorithm:
        raise RuntimeError("Milvus created the sparse index with an unexpected algorithm.")

    stats = client.get_collection_stats(collection_name)
    if int(stats["row_count"]) != indexed_document_count:
        raise RuntimeError(
            "Milvus indexed a different number of documents than were submitted: "
            f"{stats['row_count']} != {indexed_document_count}."
        )

    client.load_collection(collection_name)
    wait_for_load(client, collection_name)
    return MilvusIndex(
        collection_name,
        document_count,
        indexed_document_count,
        algorithm,
    )


def retrieve(client, index, query_embeddings, k, batch_size, drop_ratio_search):
    if k < 1:
        raise ValueError("k must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if not 0 <= drop_ratio_search < 1:
        raise ValueError("Search drop ratio must be at least 0 and less than 1.")

    stats = client.get_collection_stats(index.collection_name)
    if int(stats["row_count"]) != index.indexed_document_count:
        raise ValueError(
            "Milvus index metadata is inconsistent with the collection row count."
        )
    description = client.describe_index(index.collection_name, SPARSE_VECTOR_FIELD)
    if (
        description.get("metric_type") != "IP"
        or description.get("inverted_index_algo") != index.algorithm
    ):
        raise ValueError("Milvus index metadata is inconsistent with the configured index.")

    depth = min(k, index.indexed_document_count)
    for query_batch in batched(query_embeddings, batch_size):
        rankings = [[] for _ in query_batch]
        vectors = []
        vector_positions = []

        if depth:
            for position, (_, tokens, values) in enumerate(query_batch):
                vector = to_sparse_vector(tokens, values)
                if not vector:
                    continue
                vectors.append(vector)
                vector_positions.append(position)

        if vectors:
            responses = client.search(
                index.collection_name,
                data=vectors,
                anns_field=SPARSE_VECTOR_FIELD,
                limit=depth,
                output_fields=[DOCUMENT_ID_FIELD],
                search_params={
                    "metric_type": "IP",
                    "params": {"drop_ratio_search": drop_ratio_search},
                },
            )
            for position, response in zip(vector_positions, responses):
                ranking = []
                for hit in response:
                    document_id = hit.get("entity", {}).get(DOCUMENT_ID_FIELD)
                    score = float(hit["distance"])
                    if document_id is None or score <= 0:
                        continue
                    ranking.append((str(document_id), score))
                rankings[position] = sorted(
                    ranking,
                    key=lambda result: result[1],
                    reverse=True,
                )[:depth]

        for (query_id, _, _), ranking in zip(query_batch, rankings):
            yield str(query_id), ranking


def read_server_log(log_path):
    try:
        return log_path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


@contextmanager
def milvus_server(storage_path, startup_timeout=180):
    storage_path = Path(storage_path)
    log_path = storage_path / "milvus.log"
    etcd_config_path = storage_path / "etcd.yaml"
    etcd_config_path.write_text(
        "\n".join(
            [
                "name: default",
                "listen-peer-urls: http://127.0.0.1:2380",
                "listen-client-urls: http://127.0.0.1:2379",
                "initial-advertise-peer-urls: http://127.0.0.1:2380",
                "advertise-client-urls: http://127.0.0.1:2379",
                "initial-cluster: default=http://127.0.0.1:2380",
                "initial-cluster-state: new",
                "",
            ]
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "DEPLOY_MODE": "STANDALONE",
            "ETCD_USE_EMBED": "true",
            "ETCD_CONFIG_PATH": str(etcd_config_path),
            "ETCD_DATA_DIR": str(storage_path / "etcd"),
            "ETCD_ENDPOINTS": "127.0.0.1:2379",
            "COMMON_STORAGETYPE": "local",
            "LOCALSTORAGE_PATH": str(storage_path / "data"),
            "ROCKSMQ_PATH": str(storage_path / "rocksmq"),
            "LOG_FILE_ROOTPATH": str(storage_path / "logs"),
            "MIXCOORD_PORT": "19531",
        }
    )

    with log_path.open("w") as server_log:
        process = subprocess.Popen(  # noqa: S603
            [MILVUS_BINARY, "run", "standalone"],
            cwd="/milvus",
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )

    client = None
    deadline = time.monotonic() + startup_timeout
    try:
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Milvus exited during startup with code {exit_code}.\n"
                    f"{read_server_log(log_path)}"
                )
            try:
                client = MilvusClient(MILVUS_URI, timeout=10)
                client.list_collections()
                break
            except MilvusException:
                if client is not None:
                    client.close()
                    client = None
                time.sleep(0.5)
        else:
            raise TimeoutError(
                f"Milvus did not start within {startup_timeout} seconds.\n"
                f"{read_server_log(log_path)}"
            )
        yield client
    finally:
        if client is not None:
            client.close()
        if process.poll() is None:
            process.kill()
            process.wait()


@retrieve_command()
@click.option(
    "--algorithm",
    type=click.Choice(
        ["DAAT_MAXSCORE", "DAAT_WAND", "TAAT_NAIVE"],
        case_sensitive=False,
    ),
    default="DAAT_MAXSCORE",
    show_default=True,
    help="Milvus sparse inverted-index query algorithm.",
)
@click.option(
    "--index-batch-size",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
    help="Number of documents sent to Milvus per insert.",
)
@click.option(
    "--query-batch-size",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="Number of queries sent to Milvus per search request.",
)
@click.option(
    "--drop-ratio-build",
    type=click.FloatRange(min=0, max=1, max_open=True),
    default=0.0,
    show_default=True,
    help="Fraction of the smallest document values omitted during indexing.",
)
@click.option(
    "--drop-ratio-search",
    type=click.FloatRange(min=0, max=1, max_open=True),
    default=0.0,
    show_default=True,
    help="Fraction of the smallest query values omitted during retrieval.",
)
def main(
    dataset,
    embedding,
    output,
    k,
    algorithm,
    index_batch_size,
    query_batch_size,
    drop_ratio_build,
    drop_ratio_search,
):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    algorithm = algorithm.upper()
    register_metadata(
        {
            "actor": {"team": "reneuir-baselines"},
            "tag": (
                f"milvus-{embedding.replace('/', '-')}-{algorithm}-"
                f"{index_batch_size}-{query_batch_size}-{drop_ratio_build}-"
                f"{drop_ratio_search}-{k}"
            ),
        }
    )

    document_embeddings = load_embeddings(dataset, embedding, "doc")
    query_embeddings = load_embeddings(dataset, embedding, "query")

    with TemporaryDirectory() as temporary_directory:
        with milvus_server(temporary_directory) as client:
            with tracking(
                export_file_path=output / "index-metadata.yml",
                export_format=ExportFormat.IR_METADATA,
            ):
                index = build_index(
                    client,
                    document_embeddings,
                    index_batch_size,
                    algorithm,
                    drop_ratio_build,
                )

            with tracking(
                export_file_path=output / "retrieval-metadata.yml",
                export_format=ExportFormat.IR_METADATA,
            ):
                results = list(
                    retrieve(
                        client,
                        index,
                        query_embeddings,
                        k,
                        query_batch_size,
                        drop_ratio_search,
                    )
                )

    with gzip.open(output / "run.txt.gz", "wt") as run_file:
        for query_id, ranking in results:
            for rank, (document_id, score) in enumerate(ranking, start=1):
                run_file.write(
                    f"{query_id} Q0 {document_id} {rank} {score} milvus\n"
                )


if __name__ == "__main__":
    main()
