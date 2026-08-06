#!/usr/bin/env python3
import gzip
import math
import os
import socket
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings


COLLECTION_NAME = "lsr-benchmark"
SPARSE_VECTOR_NAME = "embedding"
QDRANT_BINARY = "/usr/local/bin/qdrant"
MAX_SPARSE_INDEX = (1 << 32) - 1


@dataclass(frozen=True)
class QdrantIndex:
    collection_name: str
    document_count: int
    indexed_document_count: int


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

    entries = sorted((index, value) for index, value in merged.items() if value != 0)
    return models.SparseVector(
        indices=[index for index, _ in entries],
        values=[value for _, value in entries],
    )


def batched(items, batch_size):
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def wait_for_collection(client, collection_name, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        collection = client.get_collection(collection_name)
        if collection.status == models.CollectionStatus.GREEN:
            return collection
        if collection.status == models.CollectionStatus.RED:
            raise RuntimeError(f"Qdrant collection {collection_name!r} entered the red state.")
        time.sleep(0.1)
    raise TimeoutError(f"Qdrant collection {collection_name!r} was not ready within {timeout} seconds.")


def build_index(
    client,
    document_embeddings,
    batch_size,
    on_disk,
    collection_name=COLLECTION_NAME,
):
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    full_scan_threshold=0,
                    on_disk=on_disk,
                    datatype=models.Datatype.FLOAT32,
                ),
                modifier=models.Modifier.NONE,
            )
        },
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )

    seen_document_ids = set()
    pending_points = []
    document_count = 0
    indexed_document_count = 0

    for point_id, (document_id, tokens, values) in enumerate(document_embeddings):
        document_id = str(document_id)
        if document_id in seen_document_ids:
            raise ValueError(f"Document IDs must be unique; found duplicate {document_id!r}.")
        seen_document_ids.add(document_id)
        document_count += 1

        vector = to_sparse_vector(tokens, values)
        if not vector.indices:
            continue
        pending_points.append(
            models.PointStruct(
                id=point_id,
                payload={"doc_id": document_id},
                vector={SPARSE_VECTOR_NAME: vector},
            )
        )
        indexed_document_count += 1
        if len(pending_points) == batch_size:
            client.upsert(collection_name, pending_points, wait=True)
            pending_points = []

    if pending_points:
        client.upsert(collection_name, pending_points, wait=True)

    collection = wait_for_collection(client, collection_name)
    if collection.points_count != indexed_document_count:
        raise RuntimeError(
            "Qdrant indexed a different number of documents than were submitted: "
            f"{collection.points_count} != {indexed_document_count}."
        )
    return QdrantIndex(collection_name, document_count, indexed_document_count)


def retrieve(client, index, query_embeddings, k, batch_size):
    if k < 1:
        raise ValueError("k must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    collection = client.get_collection(index.collection_name)
    if collection.points_count != index.indexed_document_count:
        raise ValueError(
            "Qdrant index metadata is inconsistent with the collection point count."
        )

    depth = min(k, index.indexed_document_count)
    for query_batch in batched(query_embeddings, batch_size):
        rankings = [[] for _ in query_batch]
        requests = []
        request_positions = []

        if depth:
            for position, (_, tokens, values) in enumerate(query_batch):
                vector = to_sparse_vector(tokens, values)
                if not vector.indices:
                    continue
                requests.append(
                    models.QueryRequest(
                        query=vector,
                        using=SPARSE_VECTOR_NAME,
                        params=models.SearchParams(exact=True),
                        limit=depth,
                        with_payload=["doc_id"],
                    )
                )
                request_positions.append(position)

        if requests:
            responses = client.query_batch_points(index.collection_name, requests)
            for position, response in zip(request_positions, responses):
                ranking = []
                for point in response.points:
                    document_id = (point.payload or {}).get("doc_id")
                    score = float(point.score)
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


def available_ports(count):
    sockets = []
    try:
        for _ in range(count):
            server_socket = socket.socket()
            server_socket.bind(("127.0.0.1", 0))
            sockets.append(server_socket)
        return [server_socket.getsockname()[1] for server_socket in sockets]
    finally:
        for server_socket in sockets:
            server_socket.close()


def read_server_log(log_path):
    try:
        return log_path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


@contextmanager
def qdrant_server(storage_path, startup_timeout=30):
    storage_path = Path(storage_path)
    http_port, grpc_port = available_ports(2)
    log_path = storage_path / "qdrant.log"
    environment = os.environ.copy()
    environment.update(
        {
            "QDRANT__STORAGE__STORAGE_PATH": str(storage_path / "storage"),
            "QDRANT__STORAGE__SNAPSHOTS_PATH": str(storage_path / "snapshots"),
            "QDRANT__SERVICE__HOST": "127.0.0.1",
            "QDRANT__SERVICE__HTTP_PORT": str(http_port),
            "QDRANT__SERVICE__GRPC_PORT": str(grpc_port),
            "QDRANT__TELEMETRY_DISABLED": "true",
        }
    )

    with log_path.open("w") as server_log:
        process = subprocess.Popen(  # noqa: S603
            [QDRANT_BINARY],
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )

    client = QdrantClient(url=f"http://127.0.0.1:{http_port}", timeout=60)
    deadline = time.monotonic() + startup_timeout
    try:
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Qdrant exited during startup with code {exit_code}.\n"
                    f"{read_server_log(log_path)}"
                )
            try:
                client.get_collections()
                break
            except (ResponseHandlingException, UnexpectedResponse):
                time.sleep(0.1)
        else:
            raise TimeoutError(
                f"Qdrant did not start within {startup_timeout} seconds.\n"
                f"{read_server_log(log_path)}"
            )
        yield client
    finally:
        client.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@retrieve_command()
@click.option(
    "--index-batch-size",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
    help="Number of documents sent to Qdrant per upsert.",
)
@click.option(
    "--query-batch-size",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="Number of queries sent to Qdrant per batch request.",
)
@click.option(
    "--on-disk/--in-memory",
    default=False,
    show_default=True,
    help="Store the sparse inverted index on disk instead of RAM.",
)
def main(
    dataset,
    embedding,
    output,
    k,
    index_batch_size,
    query_batch_size,
    on_disk,
):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    register_metadata(
        {
            "actor": {"team": "reneuir-baselines"},
            "tag": (
                f"qdrant-{embedding.replace('/', '-')}-{index_batch_size}-"
                f"{query_batch_size}-{on_disk}-{k}"
            ),
        }
    )

    document_embeddings = load_embeddings(dataset, embedding, "doc")
    query_embeddings = load_embeddings(dataset, embedding, "query")

    with TemporaryDirectory() as temporary_directory:
        with qdrant_server(temporary_directory) as client:
            with tracking(
                export_file_path=output / "index-metadata.yml",
                export_format=ExportFormat.IR_METADATA,
            ):
                index = build_index(
                    client,
                    document_embeddings,
                    index_batch_size,
                    on_disk,
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
                    )
                )

    with gzip.open(output / "run.txt.gz", "wt") as run_file:
        for query_id, ranking in results:
            for rank, (document_id, score) in enumerate(ranking, start=1):
                run_file.write(
                    f"{query_id} Q0 {document_id} {rank} {score} qdrant\n"
                )


if __name__ == "__main__":
    main()
