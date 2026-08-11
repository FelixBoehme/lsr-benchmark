from pathlib import Path

import yaml

from lsr_benchmark.datasets import all_dense_embeddings


def download_embeddings(embedding: str | Path, dataset: str, tira) -> Path:
    emb_is_path = isinstance(embedding, Path)
    if emb_is_path:
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
        meta_file = embeddings_dir / folder / f"{folder}-ir-metadata.yml"
        with open(meta_file, "r") as f:
            meta = yaml.safe_load(f)

        if "data" in meta:
            if "test collection" in meta["data"] and "name" in meta["data"]["test collection"]:
                meta["data"]["test collection"]["name"] = dataset
            if not emb_is_path:
                meta["data"]["embedding model"] = {"name": embedding}

            with open(meta_file, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    return embeddings_dir
