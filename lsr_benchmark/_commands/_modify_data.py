import click
import json
import gzip
import numpy as np

from lsr_benchmark.datasets import all_embeddings, all_dense_embeddings
from tira.rest_api_client import Client
from tira.third_party_integrations import default_tira_cache_dir
from pathlib import Path

class DependentOption(click.Option):
    def __init__(self, *args, requires=None, **kwargs):
        self.requires = requires
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.name in opts and not ctx.params.get(self.requires):
            raise click.UsageError(f"--{self.name.replace('_', '-')} requires --{self.requires.replace('_', '-')}")
        return super().handle_parse_result(ctx, opts, args)

# TODO: add remaining datasets, add test to check if dataset is missing from map
DATSET_TO_MAPPING = {
    "tiny-example-20251002_0-training": "d1",
    "trec-28-deep-learning-passages-20250926-training": "d2",
}

def map_dataset(dataset):
    try:
        return DATSET_TO_MAPPING[dataset]
    except KeyError as e:
        choices = ", ".join([f"'{k}'" for k in DATSET_TO_MAPPING.keys()])
        raise ValueError(f"{e} not one of {choices}")

def get_embedding_path(embedding, dataset_id, tira):
    if isinstance(embedding, Path):
        embeddings_dir = embedding.resolve()
    elif embedding.lower() != "none" and embedding not in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/lightning-ir/{embedding}', dataset_id)
    elif embedding.lower() != "none" and embedding in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/sentence-transformers/{embedding}', dataset_id)
    else:
        embeddings_dir = None
    return embeddings_dir

def add_prefixes(file, out, prefix, field):
    for line in file:
        if line.strip():
            query = json.loads(line)
            query[field] = f"{prefix}-{field}"
            out.write(json.dumps(query) + "\n")

def pack_4bit(arr: np.array):
    if len(arr) % 2 != 0:
        arr = np.append(arr, 0)
    pairs = arr.reshape(-1, 2)
    packed = (pairs[:, 0] << 4) | pairs[:, 1]
    return packed.astype(np.uint8)


def pack_2bit(arr):
    remainder = len(arr) % 4
    if remainder != 0:
        arr = np.append(arr, np.zeros(4 - remainder, dtype=np.uint8))
    groups = arr.reshape(-1, 4)
    packed = (groups[:, 0] << 6) | (groups[:, 1] << 4) | (groups[:, 2] << 2) | (groups[:, 3])
    return packed.astype(np.uint8)

def quantize(embeddings: np.ndarray, level: int, bitpack: bool):
    match level:
        case 1 | 2 | 4:
            normalized = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
            quantized = np.round(normalized * (2**level - 1)).astype(np.int8)
            if not bitpack:
                return quantized
            else:
                if level == 1:
                    return np.packbits(quantized)
                elif level == 2:
                    return pack_2bit(quantized)
                else:
                    return pack_4bit(quantized)
        case 8:
            return (embeddings * 255).astype(np.int8)
        case 16:
            return embeddings.astype(np.float16)
        case _:
            raise ValueError(f"Quantizing to {level} bits is not supported.")

@click.argument(
    "datasets",
    type=str,
    nargs=-1
)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings() + list(all_dense_embeddings())),
    required=True,
    # TODO: enable use for multiple embeddings
    # multiple=True,
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
    type=int,
    # TODO: allow multiple levels
    help="Number of bits to quantize data to"
)
@click.option(
    "-b",
    "--bitpack",
    cls=DependentOption,
    requires="quantization",
    is_flag=True,
    help="Whether to bitpack the quantized data"
)
def modify_data(datasets: list[str], embedding: str, join: bool, quantization: int, bitpack: bool) -> int:
    if not (join or quantize):
        raise ValueError("No modification chosen! Aborting.")
    for d in datasets:
        if d not in DATSET_TO_MAPPING:
            choices = ", ".join([f"'{k}'" for k in DATSET_TO_MAPPING.keys()])
            raise ValueError(f"{d} not one of {choices}")
    tira = Client()
    datasets = [
        (
            DATSET_TO_MAPPING[d],
            tira.download_dataset("lsr-benchmark", d),
            get_embedding_path(embedding, d, tira)
        )
        for d in datasets
    ]
    joint_mappings = "-".join(sorted([d[0] for d in datasets]))
    tira_dir = default_tira_cache_dir()

    result_path = Path(f"{tira_dir}/extracted_datasets/lsr-benchmark/{joint_mappings}/")
    result_path.mkdir(exist_ok=True, parents=True)
    with open(result_path/"queries.jsonl", "w") as out:
        for mapping, path, _ in datasets:
            with open(path/"queries.jsonl", "r") as file:
                add_prefixes(file, out, mapping, "qid")
    with gzip.open(result_path/"corpus.jsonl.gz", "wt") as out:
        for mapping, path, _ in datasets:
            with gzip.open(path/"corpus.jsonl.gz", "rt") as file:
                add_prefixes(file, out, mapping, "doc_id")

    if quantization:
        if quantization == 16:
            joint_mappings += "-fp16"
        else:
            joint_mappings += f"-q{quantization}"
    result_path = Path(f"{tira_dir}/extracted_runs/lsr-benchmark/{joint_mappings}/{embedding}")
    result_path.mkdir(parents=True, exist_ok=True)
    Path(result_path/"doc").mkdir(exist_ok=True)
    Path(result_path/"query").mkdir(exist_ok=True)
    for data_dir in ["doc/doc-ids.txt", "query/query-ids.txt"]:
        with open(result_path/data_dir, "w") as out:
            for mapping, _, path in datasets:
                with open(path/data_dir, "r") as file:
                    for line in file:
                        out.write(f"{mapping}-{line.strip()}\n")
    all_data = []
    all_indices = []
    all_indptr = []
    current_offset = 0
    for data_dir in ["doc/doc-embeddings.npz", "query/query-embeddings.npz"]:
        for i, (_, _, path) in enumerate(datasets):
            with np.load(path/data_dir) as npzFile:
                data = npzFile["data"]
                indices = npzFile["indices"]
                indptr = npzFile["indptr"]

                all_data.append(data)
                all_indices.append(indices)

                if i == 0:
                    all_indptr.append(indptr)
                else:
                    all_indptr.append(indptr[1:] + current_offset)
            current_offset += len(data)
        data = np.concatenate(all_data)
        if quantization:
            data = quantize(data, quantization, bitpack)
        np.savez_compressed(
            result_path/data_dir,
            data=data,
            indices=np.concatenate(all_indices),
            indptr=np.concatenate(all_indptr)
        )
    return 0
