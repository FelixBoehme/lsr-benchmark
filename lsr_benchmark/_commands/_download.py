import click
from tira.rest_api_client import Client
from lsr_benchmark.datasets import (
    all_embeddings, all_dense_embeddings, all_ir_datasets,
    IR_DATASET_TO_TIRA_DATASET, EMBEDDING_MODEL_TO_ENGINE
)
from shutil import copytree

from .sisap_io import MissingSisapDependencyError, export_embeddings_to_sisap


@click.option(
    "--dataset",
    type=click.Choice(all_ir_datasets()),
    required=True,
)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings() + sorted(list(all_dense_embeddings()))),
    required=True,
)
@click.option(
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default=None,
    help="The output directory to write to.",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["reneuir", "sisap"]),
    required=False,
    multiple=False,
    default="reneuir",
    help="The output format to write.",
)
def download_embeddings(dataset, embedding, out, export_format):
    tira = Client()
    engine = EMBEDDING_MODEL_TO_ENGINE.get(embedding, "lightning-ir")
    tira_dataset = IR_DATASET_TO_TIRA_DATASET[dataset]
    source_dir = Path(tira.get_run_output(f'lsr-benchmark/{engine}/{embedding}', tira_dataset))
    ret = source_dir
    if export_format == "sisap":
        if out is None:
            raise click.UsageError("--out is required when --format sisap is used.")
        try:
            ret = export_embeddings_to_sisap(source_dir, Path(out), dataset)
        except MissingSisapDependencyError as exc:
            raise click.ClickException(str(exc)) from exc
    elif out is not None:
        copytree(ret, out)
        ret = out
    print(ret)


@click.option(
    "--dataset",
    type=click.Choice(all_ir_datasets()),
    required=True,
)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings() + sorted(list(all_dense_embeddings()))),
    required=True,
)
@click.option(
    "--retrieval",
    type=click.Choice(sorted(["seismic", "duckdb", "kannolo", "naive-search",
                               "pyterrier-splade-pisa", "pyterrier-splade",
                               "pytorch-naive", "numpy-exhaustive"])),
    required=True,
)
@click.option(
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default=None,
    help="The output directory to write to.",
)
def download_run(dataset, embedding, retrieval, out):
    tira = Client()
    system_name = f'lsr-benchmark/reneuir-baselines/{retrieval}-on-{embedding.replace("/", "-")}'
    tira_dataset = IR_DATASET_TO_TIRA_DATASET[dataset]
    ret = tira.get_run_output(system_name, tira_dataset)
    if out is not None:
        copytree(ret, out)
        ret = out
    print(ret)
