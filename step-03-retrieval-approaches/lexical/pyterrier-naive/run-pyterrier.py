#!/usr/bin/env python3
import gzip
from pathlib import Path
from shutil import rmtree

import click
import ir_datasets
import pandas as pd
import pyterrier as pt
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import option_lsr_dataset, option_retrieval_depth


@click.command()
@option_lsr_dataset()
@option_retrieval_depth()
@click.option(
    "--index",
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    required=True,
    help="The path of the index to use.",
)
@click.option(
    "--retrieval",
    type=click.Choice(["BM25", "DPH", "PL2", "DIRICHLET_LM", "HIEMSTRA_LM", "TF", "TF_IDF"]),
    required=False,
    default="BM25",
    help="The retrieval model to use.",
)
def main(dataset, output, index, retrieval, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    ensure_pyterrier_is_loaded(boot_packages=())

    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": f"pyterrier-naive-{retrieval.lower()}-top-{k}"})

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        index = pt.terrier.TerrierIndex(index / "doc" / "doc-index")

    rmtree(output / ".tirex-tracker")
    queries = []

    for i in ir_dataset.queries_iter():
        queries.extend([{"qid": i.query_id, "query": i.default_text()}])

    pipeline = pt.terrier.Retriever(index, wmodel=retrieval, num_results=k)
    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        run = pipeline(pd.DataFrame(queries))

    rmtree(output / ".tirex-tracker")
    run["rank"] += 1
    with gzip.open(output / "run.txt.gz", "wt") as f:
        for qid, _, docid, docno, rank, score in run.itertuples(index=False):
            f.write(f"{qid} Q0 {docno} {rank} {score} {retrieval}\n")


if __name__ == "__main__":
    main()
