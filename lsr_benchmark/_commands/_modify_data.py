import gzip
import json
import shutil
from pathlib import Path

import click
import numpy as np
from tira.rest_api_client import Client
from tira.third_party_integrations import default_tira_cache_dir

from lsr_benchmark.datasets import all_dense_embeddings, all_embeddings

DATASET_TO_MAPPING = {
    "tiny-example-20251002_0-training": "d1",
    "trec-18-web-20251008-test": "d2",
    "trec-19-web-20251008-test": "d3",
    "trec-20-web-20251008-test": "d4",
    "trec-21-web-20251008-test": "d5",
    "trec-22-web-20251008-test": "d6",
    "trec-23-web-20251008-test": "d7",
    "trec-28-deep-learning-passages-20250926-training": "d8",
    "trec-28-misinfo-20251008_1-test": "d9",
    "trec-29-deep-learning-passages-20250926-training": "d10",
    "trec-33-rag-20250926_1-training": "d11",
    "trec-robust-2004-fold-1-20250927-test": "d12",
    "trec-robust-2004-fold-2-20250926-test": "d13",
    "trec-robust-2004-fold-3-20250926-test": "d14",
    "trec-robust-2004-fold-4-20250926-test": "d15",
    "trec-robust-2004-fold-5-20250926-test": "d16",
}


def get_embedding_path(embedding: str, dataset_id: str, tira: Client) -> Path | None:
    if embedding.lower() != "none" and embedding not in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f"lsr-benchmark/lightning-ir/{embedding}", dataset_id)
    elif embedding.lower() != "none" and embedding in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f"lsr-benchmark/sentence-transformers/{embedding}", dataset_id)
    else:
        embeddings_dir = None
    return embeddings_dir


def prefix_json(file, out, prefix: str, field: str) -> None:
    for line in file:
        if line.strip():
            record = json.loads(line)
            record[field] = f"{prefix}-{record[field]}"
            out.write(json.dumps(record) + "\n")


def quantize(embeddings: np.ndarray, level: int) -> np.ndarray:
    match level:
        case 1 | 2 | 4:
            normalized = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
            quantized = np.round(normalized * (2**level - 1)).astype(np.int8)
            return quantized
        case 8:
            return (embeddings * 255).astype(np.uint8)
        case 16:
            return embeddings.astype(np.float16)
        case _:
            raise ValueError(f"Quantizing to {level} bits is not supported.")


def load_and_merge_embeddings(
    embedding_paths: list[Path],
    data_dir: str,
) -> dict[str, np.ndarray]:
    all_data = []
    all_indices = []
    all_indptr = []
    current_offset = 0

    for i, emb_path in enumerate(embedding_paths):
        with np.load(emb_path / data_dir) as npz:
            data = npz["data"]
            indices = npz["indices"]
            indptr = npz["indptr"]

            all_data.append(data)
            all_indices.append(indices)
            all_indptr.append(indptr if i == 0 else indptr[1:] + current_offset)
            current_offset += len(data)

    return {
        "data": np.concatenate(all_data),
        "indices": np.concatenate(all_indices),
        "indptr": np.concatenate(all_indptr),
    }

@click.argument(
    "datasets",
    type=click.Choice(list(DATASET_TO_MAPPING.keys())),
    nargs=-1
)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings() + list(all_dense_embeddings())),
    required=True,
    multiple=True,
    help="The embeddings to run on"
)
@click.option(
    "-j",
    "--join",
    is_flag=True
)
@click.option(
    "-q",
    "--quantization",
    type=click.Choice([1, 2, 4, 8, 16]),
    multiple=True,
    help="Number of bits to quantize data to"
)
def modify_data(datasets: list[str], embedding: list[str], join: bool, quantization: list[int]) -> int:
    if not join and not quantization:
        raise click.UsageError("No modification chosen! Aborting.")

    tira = Client()
    mappings = [DATASET_TO_MAPPING[d] for d in datasets]
    joint_mappings = "-".join(sorted(mappings))
    tira_dir = default_tira_cache_dir()

    dataset_paths = [tira.download_dataset("lsr-benchmark", d) for d in datasets]

    if join:
        join_path = Path(f"{tira_dir}/extracted_datasets/lsr-benchmark/{joint_mappings}/")
        join_path.mkdir(exist_ok=True, parents=True)

        with open(join_path / "queries.jsonl", "w") as out:
            for mapping, path in zip(mappings, dataset_paths):
                with open(path / "queries.jsonl", "r") as file:
                    prefix_json(file, out, mapping, "qid")
        with gzip.open(join_path / "corpus.jsonl.gz", "wt") as out:
            for mapping, path in zip(mappings, dataset_paths):
                with gzip.open(path / "corpus.jsonl.gz", "rt") as file:
                    prefix_json(file, out, mapping, "doc_id")

    for emb in embedding:
        embedding_paths = [get_embedding_path(emb, d, tira) for d in datasets]

        if join:
            for emb_file in ["doc/doc-embeddings.npz", "query/query-embeddings.npz"]:
                merged_embeddings = load_and_merge_embeddings(embedding_paths, emb_file)

                for quant_level in quantization or [None]:
                    suffix = "-fp16" if quant_level == 16 else f"-q{quant_level}" if quant_level is not None else ""
                    emb_result_path = Path(f"{tira_dir}/extracted_runs/lsr-benchmark/{joint_mappings}{suffix}/{emb}")
                    (emb_result_path / "doc").mkdir(parents=True, exist_ok=True)
                    (emb_result_path / "query").mkdir(exist_ok=True)

                    np.savez_compressed(
                        emb_result_path / emb_file,
                        data=quantize(merged_embeddings["data"], quant_level)
                        if quant_level
                        else merged_embeddings["data"],
                        indices=merged_embeddings["indices"],
                        indptr=merged_embeddings["indptr"],
                    )

                    for id_file in ["doc/doc-ids.txt", "query/query-ids.txt"]:
                        with open(emb_result_path / id_file, "w") as out:
                            for mapping, path in zip(mappings, embedding_paths):
                                with open(path / id_file, "r") as file:
                                    for line in file:
                                        out.write(f"{mapping}-{line.strip()}\n")
        elif quantization:
            for dataset, emb_path in zip(datasets, embedding_paths):
                for emb_file in ["doc/doc-embeddings.npz", "query/query-embeddings.npz"]:
                    data = load_and_merge_embeddings([emb_path], emb_file)

                    for quant_level in quantization:
                        suffix = "-fp16" if quant_level == 16 else f"-q{quant_level}"
                        emb_result_path = Path(f"{tira_dir}/extracted_runs/lsr-benchmark/{dataset}{suffix}/{emb}")
                        (emb_result_path / "doc").mkdir(parents=True, exist_ok=True)
                        (emb_result_path / "query").mkdir(exist_ok=True)

                        np.savez_compressed(
                            emb_result_path / emb_file,
                            data=quantize(data["data"], quant_level),
                            indices=data["indices"],
                            indptr=data["indptr"],
                        )

                        for id_file in ["doc/doc-ids.txt", "query/query-ids.txt"]:
                            shutil.copy(emb_path / id_file, emb_result_path / id_file)
    return 0
