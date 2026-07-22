#!/usr/bin/env python3
import gzip
from shutil import rmtree

import click
import ir_datasets
import lsr_benchmark
from guidekp import reorder_documents
from lsp import Indexer
from lsr_benchmark.click import retrieve_command
from tirex_tracker import ExportFormat, register_metadata, tracking
from tqdm import tqdm


@retrieve_command()
@click.option("--beta", type=float, required=False, default=0.33, help="Query pruning factor (fraction of query terms kept).")
@click.option("--gamma", type=int, required=False, default=250, help="Guaranteed number of superblocks to evaluate.")
@click.option("--mu", type=float, required=False, default=0.0001, help="Threshold overestimation factor (1.0 = safe).")
@click.option("--eta", type=float, required=False, default=1.0, help="Probabilistic safeness factor (1.0 = safe).")
@click.option("--bsize", type=str, required=False, default="128,8", help="Comma-separated block sizes (superblock,block).")
@click.option("--quant", type=int, required=False, default=4, help="Quantization of the block upper bounds: currently only 4 bits and 8 bits are supported.")
@click.option("--compression", type=click.Choice(["raw", "superblock", "simdbp"]), required=False, default="simdbp", help="Compression of the block upper bounds.")
@click.option("--compress-doc", type=click.Choice(["seismic", "flatinv", "bmp"]), required=False, default="seismic", help="Layout of the document index. All three rank identically; they differ in memory footprint and block-scoring speed.")
def main(dataset, embedding, output, k, beta, gamma, mu, eta, bsize, quant, compression, compress_doc):
    # LSP only stores kth-score statistics for these depths; any other k reads past
    # the end of that vector and panics inside the Rust engine.
    if k not in (10, 100, 1000):
        raise click.BadParameter("LSP supports k of 10, 100 or 1000", param_hint="--k")
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    block_sizes = [int(b) for b in bsize.split(",")]

    # the index layout is part of the tag: the variants rank identically but are
    # exactly what this task measures the efficiency of. the document ordering
    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": f"lsp-{embedding.replace('/', '-')}-{compression}-{compress_doc}-{quant}-{beta}-{gamma}-{mu}-{k}"})

    # the values in the npz files are not guaranteed to be numeric (e.g., the query
    # embeddings of some datasets store them as strings), hence the astype below
    doc_embeddings = [
        (doc_id, tokens.tolist(), values.astype("float32").tolist())
        for doc_id, tokens, values in ir_dataset.doc_embeddings(model_name=embedding)
    ]

    # LSP quantizes document term weights to 8-bit impacts relative to the global maximum.
    max_value = max((max(values) for _, _, values in doc_embeddings if values), default=1.0)

    # Reorder documents with GuideKP before indexing. The LSP index lays documents out in
    # blocks in insertion order, so permuting the list here is the document reordering.
    # GuideKP normally operates on CIFF; reorder_documents builds its forward index directly
    # from these sparse vectors and returns the new order (order[new_pos] = original index).
    # It is given the same max_value as the index so its weight->tf quantization matches.
    order = reorder_documents(
        [doc_id for doc_id, _, _ in doc_embeddings],
        [tokens for _, tokens, _ in doc_embeddings],
        [values for _, _, values in doc_embeddings],
        max_value=max_value,
    )
    doc_embeddings = [doc_embeddings[i] for i in order]

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        indexer = Indexer(bsize=block_sizes, max_value=max_value, quant=quant, compress_range=compression, compress_doc=compress_doc)
        for doc_id, tokens, values in tqdm(doc_embeddings, "create lsp index"):
            indexer.add_document(doc_id, tokens, values)
        searcher = indexer.build()

    query_embeddings = list(ir_dataset.query_embeddings(model_name=embedding))

    rmtree(output / ".tirex-tracker")

    results = []

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        for _, tokens, values in query_embeddings:
            results.append(searcher.search(
                tokens.tolist(), values.astype("float32").tolist(),
                k=k, beta=beta, gamma=gamma, mu=mu, eta=eta,
            ))

    rmtree(output / ".tirex-tracker")
    with gzip.open(output / "run.txt.gz", "wt") as f:
        for (query_id, _, _), (docnos, scores) in zip(query_embeddings, results):
            for rank, (docno, score) in enumerate(zip(docnos, scores), start=1):
                f.write(f"{query_id} Q0 {docno} {rank} {score} lsp\n")


if __name__ == "__main__":
    main()
