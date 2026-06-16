#!/usr/bin/env python3
import gzip
from shutil import rmtree

import click
import ir_datasets
import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from tirex_tracker import ExportFormat, register_metadata, tracking
from tqdm import tqdm

from sqlite_retrieval import create_connection, index_documents, retrieve_query


def remove_tracker_directory(output):
    tracker_directory = output / ".tirex-tracker"
    if tracker_directory.exists():
        rmtree(tracker_directory)


@retrieve_command()
@click.option("--quantize", is_flag=True, help="Whether to quantize the index scores to integers.")
def main(dataset, embedding, output, quantize, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    register_metadata({
        "actor": {"team": "reneuir-baselines"},
        "tag": f"sqlite-{embedding.replace('/', '-')}-{'quantize-' if quantize else ''}{k}",
    })

    conn = create_connection()

    print("Indexing documents in SQLite..")
    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        index_documents(
            conn,
            tqdm(ir_dataset.doc_embeddings(model_name=embedding), "index documents in sqlite"),
            quantize=quantize,
        )

    remove_tracker_directory(output)
    results = []

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        for query_id, tokens, values in tqdm(ir_dataset.query_embeddings(model_name=embedding), "retrieve with sqlite"):
            results.append(retrieve_query(conn, query_id, tokens, values, k))

    remove_tracker_directory(output)
    with gzip.open(output / "run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            rank = 1
            for qid, score, docno in ranking_for_query:
                f.write(f"{qid} Q0 {docno} {rank} {score} sqlite\n")
                rank += 1

    conn.close()


if __name__ == "__main__":
    main()
