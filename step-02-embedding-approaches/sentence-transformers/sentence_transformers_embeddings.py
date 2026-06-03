#!/usr/bin/env python3
from pathlib import Path
import click
import numpy as np
from sentence_transformers import SentenceTransformer
from tirex_tracker import register_metadata, tracking
import lsr_benchmark
from lsr_benchmark.click import option_lsr_dataset
import ir_datasets

TRUNCATE_LENGTH = 512


def convert_embeddings_dense(embeddings: np.ndarray):
    n_docs, n_dims = embeddings.shape
    row_idcs = np.repeat(np.arange(n_docs), n_dims)
    col_idcs = np.tile(np.arange(n_dims), n_docs)
    values = embeddings.flatten()
    row_indices = np.bincount(row_idcs + 1, minlength=n_docs + 1).cumsum()
    return values, col_idcs, row_indices


def truncate_texts(texts):
    ret = []
    for t in texts:
        # an guess on the upper bound for transformer tokens is 10 characters per token.
        ret.append(t[:(10*TRUNCATE_LENGTH)])
    return ret


def embedd_text_with_model(model, texts, ids, output):
    output.parent.mkdir(parents=True, exist_ok=True)

    # not all models use the passed module.tokenizer.model_max_length
    texts = truncate_texts(texts)

    with tracking(export_file_path=str(output).replace("-embeddings.npz", "-ir-metadata.yml")):
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    data, indices, indptr = convert_embeddings_dense(embeddings)
    np.savez_compressed(output, data=data, indices=indices, indptr=indptr)
    Path(str(output).replace("-embeddings.npz", "-ids.txt")).write_text("\n".join(ids))


@click.command()
@option_lsr_dataset()
@click.option("--model", type=str, required=True, help="The Sentence Transformers model.")
@click.option("--batch_size", type=int, default=32, help="Batch size.")
def main(dataset: str, model: str, batch_size: int, output: Path):
    lsr_benchmark.register_to_ir_datasets(dataset)
    module = SentenceTransformer(model)
    module.tokenizer.model_max_length = TRUNCATE_LENGTH
    register_metadata({"actor": {"team": "sentence-transformers"}, "tag": model.replace('/', '-')})

    dataset_id = f"lsr-benchmark/{dataset}"
    ir_dataset = ir_datasets.load(dataset_id)

    texts = [q.default_text() for q in ir_dataset.queries_iter()]
    ids = [q.query_id for q in ir_dataset.queries_iter()]
    embedd_text_with_model(module, texts, ids, Path(output) / "query" / "query-embeddings.npz")

    texts = [d.default_text() for d in ir_dataset.docs_iter()]
    ids = [d.doc_id for d in ir_dataset.docs_iter()]
    embedd_text_with_model(module, texts, ids, Path(output) / "doc" / "doc-embeddings.npz")


if __name__ == "__main__":
    main()
