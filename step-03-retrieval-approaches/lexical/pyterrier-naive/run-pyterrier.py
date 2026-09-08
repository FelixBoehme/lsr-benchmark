#!/usr/bin/env python3
import gzip
import shutil
from pathlib import Path

import click
import ir_datasets
import pandas as pd
import pyterrier as pt
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark._commands._modify_data import JOINT_TO_DATASETS, DuplicateBehaviour
from lsr_benchmark.click import option_lsr_dataset, option_retrieval_depth


def find_index_path(path: Path) -> Path | None:
    index_path = path / "doc-index"
    if index_path.exists():
        return index_path

    return None


def dataset_id(dataset: str) -> str:
    dataset_path = Path(dataset)
    return dataset_path.name if dataset_path.is_dir() else dataset


def resolve_index_paths(index: Path, dataset: str) -> list[Path]:
    direct_index = find_index_path(index)
    if direct_index:
        return [direct_index]

    index_paths = {
        path.name: index_path
        for path in index.iterdir()
        if path.is_dir() and (index_path := find_index_path(path)) is not None
    }

    if not index_paths:
        raise click.UsageError(f"No PyTerrier index found in '{index}'.")

    dataset_path = Path(dataset)
    dataset_name = dataset_id(dataset)
    if not dataset_path.is_dir() or dataset_name not in JOINT_TO_DATASETS:
        raise click.UsageError(f"PyTerrier indexes found in joint index structure for non-joint dataset '{dataset}'.")

    individual_datasets = JOINT_TO_DATASETS[dataset_name]["datasets"]
    missing_datasets = [name for name in individual_datasets if name not in index_paths]
    if missing_datasets:
        raise click.UsageError(f"Missing PyTerrier indexes for '{dataset_name}': {', '.join(missing_datasets)}.")

    return [index_paths[name] for name in individual_datasets]


def joined_documents(index_paths: list[Path], behaviour: DuplicateBehaviour):
    seen_ids = set()

    for i, index_path in enumerate(index_paths):
        source_index = pt.terrier.TerrierIndex(index_path)
        for doc in source_index.get_corpus_iter():
            docno = doc["docno"]

            if behaviour == DuplicateBehaviour.FAIL:
                if docno in seen_ids:
                    raise ValueError(f"Duplicate docno '{docno}' found while joining indexes.")
                seen_ids.add(docno)
            elif behaviour == DuplicateBehaviour.PREFIX:
                doc["docno"] = f"d{i}-{docno}"
            elif behaviour == DuplicateBehaviour.SKIP:
                if docno in seen_ids:
                    continue
                seen_ids.add(docno)

            yield doc

        del source_index
        shutil.rmtree(index_path.parent)


def load_index(index_path: Path, dataset: str):
    index_paths = resolve_index_paths(index_path, dataset)
    if len(index_paths) == 1:
        return pt.terrier.TerrierIndex(index_paths[0])

    joined_index_path = index_path / "doc-index"
    joined_index_path.parent.mkdir(parents=True, exist_ok=True)
    index = pt.terrier.TerrierIndex(joined_index_path)
    behaviour = JOINT_TO_DATASETS[dataset_id(dataset)]["settings"].doc
    indexer = index.toks_indexer()
    indexer.setProperty("termpipelines", "Stopwords,PorterStemmer")
    indexer.index(joined_documents(index_paths, behaviour))

    return index


def write_index_metadata(index_path: Path, out_path: Path) -> None:
    if (index_path / "doc-index").is_dir():
        shutil.copy(index_path / "doc-ir-metadata.yml", out_path / "index-metadata.yml")
    else:
        # TODO: decide on how to merge the multiple metadata files
        pass


@click.command()
@option_lsr_dataset()
@option_retrieval_depth()
@click.option(
    "--index",
    "index_path",
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
def main(dataset, output, index_path, retrieval, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    ensure_pyterrier_is_loaded(boot_packages=())

    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": f"pyterrier-naive-{retrieval.lower()}-top-{k}"})

    write_index_metadata(index_path, output)
    index = load_index(index_path, dataset)

    queries = []
    for i in ir_dataset.queries_iter():
        queries.extend([{"qid": i.query_id, "query": i.default_text()}])

    pipeline = (
        pt.rewrite.tokenise(matchop=True)
        >> pt.terrier.Retriever(index, wmodel=retrieval, num_results=k)
        >> pt.rewrite.reset()
    )
    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        run = pipeline(pd.DataFrame(queries))

    shutil.rmtree(output / ".tirex-tracker")
    run["rank"] += 1
    with gzip.open(output / "run.txt.gz", "wt") as f:
        for qid, _, docid, docno, rank, score in run.itertuples(index=False):
            f.write(f"{qid} Q0 {docno} {rank} {score} {retrieval}\n")


if __name__ == "__main__":
    main()
