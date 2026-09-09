#!/usr/bin/env python3

import click
import ir_datasets
import pyterrier as pt
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tirex_tracker import ExportFormat, register_metadata, tracking
from tqdm import tqdm

import lsr_benchmark
from lsr_benchmark.click import option_lsr_dataset


@click.command()
@option_lsr_dataset()
def main(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    ensure_pyterrier_is_loaded(boot_packages=())

    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": "pyterrier-lexical-index"})
    documents = [{"docno": i.doc_id, "text": i.default_text()} for i in ir_dataset.docs_iter()]

    with tracking(export_file_path=output / "doc-ir-metadata.yml", export_format=ExportFormat.IR_METADATA):
        indexer = pt.IterDictIndexer(str((output / "doc-index").resolve()), meta={"docno": 100})
        indexer.index(tqdm(documents, "Index docs"))


if __name__ == "__main__":
    main()
