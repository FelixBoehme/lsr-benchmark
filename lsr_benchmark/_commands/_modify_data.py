import gzip
import json
from pathlib import Path

import click
import numpy as np
from tira.rest_api_client import Client
from tira.third_party_integrations import default_tira_cache_dir

from lsr_benchmark.datasets import all_dense_embeddings, all_embeddings


class DependentOption(click.Option):
    def __init__(self, *args, requires=None, **kwargs):
        self.requires = requires
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.name in opts and not ctx.params.get(self.requires):
            raise click.UsageError(f"--{self.name.replace('_', '-')} requires --{self.requires.replace('_', '-')}")
        return super().handle_parse_result(ctx, opts, args)

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

# TODO: maybe allow local path, would have to contain embeddings for all datasets
# or would need multiple passed embeddings with according dataset
def get_embedding_path(embedding: str | Path, dataset_id: str, tira: Client) -> Path | None:
    if embedding.lower() != "none" and embedding not in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/lightning-ir/{embedding}', dataset_id)
    elif embedding.lower() != "none" and embedding in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/sentence-transformers/{embedding}', dataset_id)
    else:
        embeddings_dir = None
    return embeddings_dir

def prefix_json(file, out, prefix: str, field: str) -> None:
    """Write lines from `file` to `out`, prefixing each record's `field` value with `prefix`."""
    for line in file:
        if line.strip():
            record = json.loads(line)
            record[field] = f"{prefix}-{record[field]}"
            out.write(json.dumps(record) + "\n")

def pack_4bit(arr: np.ndarray) -> np.ndarray:
    """Pack an array of 4-bit values (0–15) into uint8, two values per byte."""
    if len(arr) % 2 != 0:
        arr = np.append(arr, 0)
    pairs = arr.reshape(-1, 2)
    packed = (pairs[:, 0] << 4) | pairs[:, 1]
    return packed.astype(np.uint8)


def pack_2bit(arr: np.ndarray) -> np.ndarray:
    """Pack an array of 2-bit values (0–3) into uint8, four values per byte."""
    remainder = len(arr) % 4
    if remainder != 0:
        arr = np.append(arr, np.zeros(4 - remainder, dtype=np.uint8))
    groups = arr.reshape(-1, 4)
    packed = (groups[:, 0] << 6) | (groups[:, 1] << 4) | (groups[:, 2] << 2) | (groups[:, 3])
    return packed.astype(np.uint8)

def quantize(embeddings: np.ndarray, level: int, bitpack: bool) -> np.ndarray:
    """Quantize embeddings to `level` bits, optionally bit-packing the result."""
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

def load_and_merge_embeddings(
    datasets: list[tuple[str, Path, Path]],
    data_dir: str,
    quantization: int | None,
    bitpack: bool,
    result_path: Path,
) -> None:
    """Concatenate sparse embedding npz files across datasets, quantize if requested, and save."""
    all_data = []
    all_indices = []
    all_indptr = []
    current_offset = 0

    for i, (_, _, emb_path) in enumerate(datasets):
        with np.load(emb_path / data_dir) as npz:
            data = npz["data"]
            indices = npz["indices"]
            indptr = npz["indptr"]

            all_data.append(data)
            all_indices.append(indices)
            all_indptr.append(indptr if i == 0 else indptr[1:] + current_offset)
            current_offset += len(data)

    merged_data = np.concatenate(all_data)
    if quantization:
        merged_data = quantize(merged_data, quantization, bitpack)

    np.savez_compressed(
        result_path / data_dir,
        data=merged_data,
        indices=np.concatenate(all_indices),
        indptr=np.concatenate(all_indptr),
    )

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
    if not (join or quantization):
        raise ValueError("No modification chosen! Aborting.")

    for d in datasets:
        if d not in DATASET_TO_MAPPING:
            choices = ", ".join([f"'{k}'" for k in DATASET_TO_MAPPING.keys()])
            raise click.BadParameter(f"'{d}' not one of {choices}", param_hint="datasets")

    tira = Client()
    resolved_datasets = [
        (
            DATASET_TO_MAPPING[d],
            tira.download_dataset("lsr-benchmark", d),
            get_embedding_path(embedding, d, tira)
        )
        for d in datasets
    ]

    joint_mappings = "-".join(sorted([d[0] for d in resolved_datasets]))
    tira_dir = default_tira_cache_dir()

    if join:
        join_path = Path(f"{tira_dir}/extracted_datasets/lsr-benchmark/{joint_mappings}/")
        join_path.mkdir(exist_ok=True, parents=True)

        with open(join_path/"queries.jsonl", "w") as out:
            for mapping, path, _ in resolved_datasets:
                with open(path/"queries.jsonl", "r") as file:
                    prefix_json(file, out, mapping, "qid")
        with gzip.open(join_path/"corpus.jsonl.gz", "wt") as out:
            for mapping, path, _ in resolved_datasets:
                with gzip.open(path/"corpus.jsonl.gz", "rt") as file:
                    prefix_json(file, out, mapping, "doc_id")

    # TODO: handle multiple datasets to be only quantized
    # TODO: handle naming when only quantizing
    suffix = (
        "-fp16" if quantization == 16
        else f"-q{quantization}" if quantization
        else ""
    )

    emb_result_path = Path(f"{tira_dir}/extracted_runs/lsr-benchmark/{joint_mappings}{suffix}/{embedding}")
    (emb_result_path/"doc").mkdir(parents=True, exist_ok=True)
    (emb_result_path/"query").mkdir(parents=True, exist_ok=True)

    for id_file in ["doc/doc-ids.txt", "query/query-ids.txt"]:
        with open(emb_result_path/id_file, "w") as out:
            for mapping, _, path in resolved_datasets:
                with open(path/id_file, "r") as file:
                    for line in file:
                        out.write(f"{mapping}-{line.strip()}\n")

    for emb_file in ["doc/doc-embeddings.npz", "query/query-embeddings.npz"]:
        load_and_merge_embeddings(resolved_datasets, emb_file, quantization, bitpack, emb_result_path)
    return 0
