import gzip
import json
import shutil
from pathlib import Path

import click
import numpy as np
from tira.rest_api_client import Client
from tira.third_party_integrations import default_tira_cache_dir
from tqdm import tqdm

from lsr_benchmark.datasets import all_datasets, all_dense_embeddings, all_embeddings

JOINT_TO_DATASETS = {
    "msmarco-passage-trec-dl-2019+2020-judged": [
        "trec-28-deep-learning-passages-20250926-training",
        "trec-29-deep-learning-passages-20250926-training",
    ],
    "disks45-nocr-trec-robust-2004-fold1+2+3+4+5": [
        "trec-robust-2004-fold-1-20250927-test",
        "trec-robust-2004-fold-2-20250926-test",
        "trec-robust-2004-fold-3-20250926-test",
        "trec-robust-2004-fold-4-20250926-test",
        "trec-robust-2004-fold-5-20250926-test",
    ],
    "clueweb12-trec-web-2013+2014+clueweb12-b13-trec-misinfo-2019": [
        "trec-22-web-20251008-test",
        "trec-23-web-20251008-test",
        "trec-28-misinfo-20251008_1-test",
    ],
    "clueweb09-en-trec-web-2009+2010+2011+2012": [
        "clueweb09/en/trec-web-2009",
        "clueweb09/en/trec-web-2010",
        "clueweb09/en/trec-web-2011",
        "clueweb09/en/trec-web-2012",
    ],
}


def get_embedding_path(embedding: str, dataset_id: str, tira: Client) -> Path:
    if embedding in set(
        [
            "e5-mistral-7b-instruct",
            "SFR-Embedding-Mistral",
            "Linq-Embed-Mistral",
            "Octen-Embedding-8B",
            "Qwen3-Embedding-8B",
            "speed-embedding-7b-instruct",
        ]
    ):
        embeddings_dir = tira.get_run_output(f"lsr-benchmark/mteb/{embedding}", dataset_id)
    elif embedding in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f"lsr-benchmark/sentence-transformers/{embedding}", dataset_id)
    else:
        embeddings_dir = tira.get_run_output(f"lsr-benchmark/lightning-ir/{embedding}", dataset_id)

    return embeddings_dir


def prefix_json(file, out, prefix: str, field: str, desc: str = "") -> None:
    for line in tqdm(file, desc=desc, leave=False, unit=" lines"):
        if line.strip():
            record = json.loads(line)
            record[field] = f"{prefix}-{record[field]}"
            out.write(json.dumps(record) + "\n")


def load_and_merge_embeddings(
    embedding_paths: list[Path],
    data_dir: str,
) -> dict[str, np.ndarray]:
    all_data = []
    all_indices = []
    all_indptr = []
    current_offset = 0

    for i, emb_path in enumerate(tqdm(embedding_paths, desc=f"Loading {data_dir}", leave=False)):
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


def perform_dataset_join(dataset: str, tira: Client, tira_dir: str) -> Path:
    individual_datasets = JOINT_TO_DATASETS[dataset]
    mappings = [f"d{i}" for i in range(len(individual_datasets))]

    dataset_paths = [tira.download_dataset("lsr-benchmark", d) for d in tqdm(individual_datasets, desc="Datasets")]

    join_path = Path(f"{tira_dir}/extracted_datasets/lsr-benchmark/{dataset}/")
    join_path.mkdir(exist_ok=True, parents=True)

    with open(join_path / "queries.jsonl", "w") as out:
        for i, (mapping, path) in tqdm(
            enumerate(zip(mappings, dataset_paths)), total=len(mappings), desc="Joining Queries"
        ):
            with open(path / "queries.jsonl", "r") as file:
                prefix_json(file, out, mapping, "qid", desc=f"Prefixing {individual_datasets[i]}")

    with gzip.open(join_path / "corpus.jsonl.gz", "wt") as out:
        for i, (mapping, path) in tqdm(
            enumerate(zip(mappings, dataset_paths)), total=len(mappings), desc="Joining Corpora"
        ):
            with gzip.open(path / "corpus.jsonl.gz", "rt") as file:
                prefix_json(file, out, mapping, "doc_id", desc=f"Prefixing {individual_datasets[i]}")

    return join_path


def perform_embedding_join(dataset: str, embedding: str, tira: Client, tira_dir: str) -> Path:
    individual_datasets = JOINT_TO_DATASETS[dataset]
    mappings = [f"d{i}" for i in range(len(individual_datasets))]

    embedding_paths = [get_embedding_path(embedding, d, tira) for d in individual_datasets]

    emb_result_path = Path(f"{tira_dir}/extracted_runs/lsr-benchmark/{dataset}/{embedding}")
    (emb_result_path / "doc").mkdir(parents=True, exist_ok=True)
    (emb_result_path / "query").mkdir(exist_ok=True)

    for emb_file in ["doc/doc-embeddings.npz", "query/query-embeddings.npz"]:
        merged_embeddings = load_and_merge_embeddings(embedding_paths, emb_file)
        np.savez_compressed(
            emb_result_path / emb_file,
            data=merged_embeddings["data"],
            indices=merged_embeddings["indices"],
            indptr=merged_embeddings["indptr"],
        )

    for folder in ["doc", "query"]:
        id_file = f"{folder}/{folder}-ids.txt"
        with open(emb_result_path / id_file, "w") as out:
            for mapping, path in zip(mappings, embedding_paths):
                with open(path / id_file, "r") as file:
                    for line in file:
                        out.write(f"{mapping}-{line.strip()}\n")

        meta_file = f"{folder}/{folder}-ir-metadata.yml"
        meta_out_dir = emb_result_path / folder
        for mapping, path in zip(mappings, embedding_paths):
            src_meta = path / meta_file
            dest_meta = meta_out_dir / f"{mapping}-{folder}-ir-metadata.yml"
            shutil.copy(src_meta, dest_meta)

    return emb_result_path


@click.argument("datasets", type=click.Choice(list(JOINT_TO_DATASETS.keys()) + list(all_datasets())), nargs=-1)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings() + list(all_dense_embeddings())),
    required=True,
    multiple=True,
    help="The embeddings to run on",
)
@click.option("-j", "--join", is_flag=True, required=True)
def modify_data(datasets: list[str], embedding: list[str], join: bool) -> int:
    if join:
        for d in datasets:
            if d not in JOINT_TO_DATASETS:
                choices_str = ", ".join([f"'{choice}'" for choice in JOINT_TO_DATASETS.keys()])
                raise click.UsageError(f"Can't create joint dataset {d!r}.\nChoose one of {choices_str}")

    tira = Client()
    tira_dir = default_tira_cache_dir()

    created_dataset_dirs = []
    created_embedding_dirs = []

    if join:
        for dataset in tqdm(datasets, desc="Joining"):
            created_dataset_dirs.append(perform_dataset_join(dataset, tira, tira_dir))
            for emb in tqdm(embedding, desc="Processing Embeddings"):
                created_embedding_dirs.append(perform_embedding_join(dataset, emb, tira, tira_dir))

    click.echo("\nFollowing paths have been created:")
    if created_dataset_dirs:
        click.echo("Dataset:")
        for path in sorted(created_dataset_dirs):
            click.echo(f"  - {path}")

    if created_embedding_dirs:
        click.echo("Embeddings:")
        for path in sorted(created_embedding_dirs):
            click.echo(f"  - {path}")

    return 0
