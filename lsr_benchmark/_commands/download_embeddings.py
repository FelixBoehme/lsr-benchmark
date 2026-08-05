from pathlib import Path

from tira.io_utils import patch_ir_metadata

from lsr_benchmark.datasets import all_dense_embeddings


def download_embeddings(embedding: str | Path, dataset: str, tira) -> Path:
    if isinstance(embedding, Path):
        embeddings_dir = embedding.resolve()
    elif embedding.lower() != "none" and embedding not in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f"lsr-benchmark/lightning-ir/{embedding}", dataset)
    elif embedding.lower() != "none" and embedding in set(
        [
            "e5-mistral-7b-instruct",
            "SFR-Embedding-Mistral",
            "Linq-Embed-Mistral",
            "Octen-Embedding-8B",
            "Qwen3-Embedding-8B",
            "speed-embedding-7b-instruct",
        ]
    ):
        embeddings_dir = tira.get_run_output(f"lsr-benchmark/mteb/{embedding}", dataset)
    elif embedding.lower() != "none" and embedding in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f"lsr-benchmark/sentence-transformers/{embedding}", dataset)
    else:
        raise ValueError(f"Unable to download unknown embeddings {embedding!r} for dataset {dataset!r}. Aborting!")

    for folder in ["doc", "query"]:
        patch_ir_metadata(
            embeddings_dir / folder,
            {"data": {"test collection": {"name": "/tira-data/input"}}},
            {"data": {"test collection": {"name": dataset}}},
        )

    return embeddings_dir
