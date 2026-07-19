import contextlib
import click
import io
import sys
from typing import Optional
from tira.io_utils import log_message, verify_docker_installation, FormatMsgType
from tira.third_party_integrations import temporary_directory
from tira.check_format import check_format
from tira.rest_api_client import Client
from pathlib import Path
from lsr_benchmark.datasets import (
    IR_DATASET_TO_TIRA_DATASET,
    all_datasets,
    all_dense_embeddings,
    all_embeddings,
)
from lsr_benchmark._commands._verify_installation import EXAMPLE_RETRIEVAL_ENGINE
from lsr_benchmark._commands._modify_data import JOINT_TO_DATASETS
from lsr_benchmark.retrieval_suites import RETRIEVAL_SUITES
import shutil
import yaml
from tira.io_utils import docker_supported_target_platform
import os


def run_retrieval_engine(
    docker_image,
    command,
    dataset_id,
    embedding,
    output_dir=None,
    platform=None,
    cpus=None,
    memory=None,
):
    if output_dir is not None and Path(output_dir).exists():
        return
    tira = Client()
    if isinstance(dataset_id, Path):
        dataset_path = dataset_id.resolve()
        dataset_id = dataset_id.stem
    else:
        dataset_path = temporary_directory()

    if isinstance(embedding, Path):
        embeddings_dir = embedding.resolve()
    elif embedding.lower() != "none" and embedding not in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/lightning-ir/{embedding}', dataset_id)
    elif embedding.lower() != "none" and embedding in set(["e5-mistral-7b-instruct", "SFR-Embedding-Mistral", "Linq-Embed-Mistral", "Octen-Embedding-8B", "Qwen3-Embedding-8B", "speed-embedding-7b-instruct"]):
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/mteb/{embedding}', dataset_id)
    elif embedding.lower() != "none" and embedding in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/sentence-transformers/{embedding}', dataset_id)
    else:
        embeddings_dir = None
    mount_directory = None
    if embeddings_dir:
        mount_directory = {"embeddings": embeddings_dir}
    tmp_dir = temporary_directory()
    tira.local_execution.run(
        image=docker_image,
        command=command,
        input_dir=dataset_path,
        output_dir=tmp_dir,
        allow_network=False,
        input_run=embeddings_dir,
        mount_directory=mount_directory,
        platform=platform if platform else docker_supported_target_platform(),
        cpu_count=cpus,
        mem_limit=memory,
    )

    result, msg = check_format(Path(tmp_dir), ["run.txt"], {})
    if result != FormatMsgType.OK:
        print(msg)
        raise ValueError(msg)

    tag = yaml.safe_load((Path(tmp_dir) / "retrieval-metadata.yml").read_text())["tag"]

    if output_dir is not None:
        from tira.io_utils import patch_ir_metadata
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_dir, output_dir)
        patch_ir_metadata(output_dir, {"data": {"test collection": {"name": "/tira-data/input"}}}, {"data": {"test collection": {"name": dataset_id}}})
        if dataset_id in JOINT_TO_DATASETS:
            for meta_file in ["index-metadata.yml", "retrieval-metadata.yml"]:
                with open(output_dir/meta_file, "r") as f:
                    meta = yaml.safe_load(f)
                meta["data"]["test collection"]["union of subsamples"] = JOINT_TO_DATASETS[dataset_id]
                with open(output_dir/meta_file, "w") as f:
                    yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    return tag


def get_approach_to_execution(approaches, platform, embedding, print_message):
    tira = Client()
    approach_to_execution = {}
    for approach in approaches:
        if approach in EXAMPLE_RETRIEVAL_ENGINE[platform]:
            approach_to_execution[approach] = {"tag": EXAMPLE_RETRIEVAL_ENGINE[platform][approach]["image"], "command": EXAMPLE_RETRIEVAL_ENGINE[platform][approach]["command"]}
        else:
            software_def = None
            last_error = None
            candidates = [approach + "-" + platform.split("/")[1]]
            for c in candidates:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        software_def = tira.public_system_details("reneuir-baselines", c)
                    break
                except ValueError as error:
                    last_error = error

            if software_def is None and last_error is not None:
                raise last_error

            approach_to_execution[approach] = {"tag": software_def["public_image_name"], "command": software_def["command"]}

    return approach_to_execution

class ChoiceOrPath(click.ParamType):
    def __init__(self, choices):
        self.choices = tuple(choices)

    def get_metavar(self, param, ctx):
        return f"[{'|'.join(self.choices)}|PATH]"

    def convert(self, value, param, ctx):
        if value in self.choices:
            return value

        path = Path(value)
        if path.exists() and path.is_dir():
            return path

        choices_str = ", ".join([f"'{choice}'" for choice in self.choices])
        self.fail(
            f"{value!r} is not one of "
            f"{choices_str} or an existing path to a directory.",
            param,
            ctx
        )


def resolve_retrieval_configuration(suite, approaches, datasets, embeddings):
    if suite is None:
        return approaches, datasets, embeddings

    if approaches or datasets or embeddings:
        raise click.UsageError(
            "--suite cannot be combined with retrieval engines, --dataset, or --embedding."
        )

    configuration = RETRIEVAL_SUITES[suite]
    suite_datasets = []
    for dataset in configuration["datasets"]:
        if dataset == "all" or dataset in all_datasets():
            suite_datasets.append(dataset)
        elif dataset in IR_DATASET_TO_TIRA_DATASET:
            suite_datasets.append(IR_DATASET_TO_TIRA_DATASET[dataset])
        else:
            raise click.UsageError(
                f"Suite {suite!r} contains unsupported dataset {dataset!r}."
            )

    supported_embeddings = {"all", "none", *all_embeddings(), *all_dense_embeddings()}
    unsupported_embeddings = [
        embedding
        for embedding in configuration["embeddings"]
        if embedding not in supported_embeddings
    ]
    if unsupported_embeddings:
        raise click.UsageError(
            f"Suite {suite!r} contains unsupported embedding "
            f"{unsupported_embeddings[0]!r}."
        )

    return (
        tuple(configuration["retrieval_engines"]),
        tuple(suite_datasets),
        tuple(configuration["embeddings"]),
    )


@click.argument(
    "approaches",
    type=str,
    nargs=-1,
)
@click.option(
    "--suite",
    type=click.Choice(sorted(RETRIEVAL_SUITES)),
    help="A predefined retrieval suite.",
)
@click.option(
    "-o", "--out",
    type=str,
    required=True,
    multiple=False,
    help="The output directory to write to.",
)
@click.option(
    "--dataset",
    type=ChoiceOrPath(["all"] + all_datasets()),
    multiple=True,
    help="The datasets to run on.",
)
@click.option(
    "--embedding",
    type=ChoiceOrPath(["all", "none", ] + all_embeddings() + list(all_dense_embeddings())),
    multiple=True,
    help="The datasets to run on.",
)
@click.option(
    "--cpus",
    type=click.IntRange(min=1),
    metavar="CPUS",
    help="The number of CPUs used for execution.",
)
@click.option(
    "--memory",
    type=str,
    metavar="MEMORY",
    help="The memory limit.",
)
def retrieval(
    approaches: list[str],
    suite: Optional[str],
    dataset: list[str],
    embedding: list[str],
    cpus: Optional[int],
    memory: Optional[str],
    out: str,
) -> int:
    all_messages = []

    def print_message(message, level):
        all_messages.append((message, level))
        os.system("cls" if os.name == "nt" else "clear") # noqa: S605
        print(' '.join([sys.argv[0].split('/')[-1]] + sys.argv[1:]))
        for message, level in all_messages:
            log_message(message, level)

    approaches, dataset, embedding = resolve_retrieval_configuration(
        suite, approaches, dataset, embedding
    )

    for d in dataset:
        for e in embedding:
            if isinstance(d, Path) and not isinstance(e, Path):
                raise click.UsageError("A local dataset has to be used in combination with local embeddings! Aborting.")

    if dataset is None or not dataset or "all" in dataset:
        dataset = all_datasets()

    if embedding is None or not embedding or "all" in embedding:
        embedding = all_embeddings()
    if embedding and "none" in embedding:
        embedding = ["none"]

    status = verify_docker_installation()

    if status[0] != FormatMsgType.OK:
        print_message("You do not have Docker installed. Please install Docker and then run 'tira-cli verify-installation' to ensure that everyting is correctly installed.", status)

        return 1

    print_message("Your TIRA installation is valid.", FormatMsgType.OK)

    platform = docker_supported_target_platform()

    if platform not in ("linux/amd64", "linux/arm64"):
        print_message(f"The platform {docker_supported_target_platform()} is not supported.", FormatMsgType.ERROR)
        return 1

    approach_to_execution = get_approach_to_execution(approaches, platform, embedding, print_message)

    if len(dataset) == 0:
        print_message("No datasets are passed.", FormatMsgType.ERROR)
        return 1

    if len(embedding) == 0:
        print_message("No embedding are passed.", FormatMsgType.ERROR)
        return 1

    if len(approaches) == 0:
        print_message("No approaches are passed.", FormatMsgType.ERROR)
        return 1

    stats = {}
    failures = 0
    for d in dataset:
        for e in embedding:
            for approach in approaches:
                dset, emb = d, e
                quant_suffix = ""
                if isinstance(d, Path):
                    dset = d.stem
                if isinstance(e, Path):
                    emb = e.stem
                out_dir = Path(out) / (dset + quant_suffix) / emb / approach
                try:
                    run_retrieval_engine(
                        approach_to_execution[approach]["tag"],
                        approach_to_execution[approach]["command"],
                        d,
                        e,
                        out_dir,
                        platform=platform,
                        cpus=cpus,
                        memory=memory,
                    )
                    if approach not in stats:
                        stats[approach] = {"embeddings": set(), "datasets": set()}
                    stats[approach]["embeddings"].add(emb)
                    stats[approach]["datasets"].add(dset)
                except Exception as error:
                    failures += 1
                    print_message(
                        f"Approach {approach} failed on dataset {dset} with "
                        f"embedding {emb}: {error}",
                        FormatMsgType.ERROR,
                    )
                    continue

    for approach in stats:
        print_message(f"Approach {approach} produced valid outputs on {len(stats[approach]['datasets'])} datasets for {len(stats[approach]['embeddings'])} embeddings.", FormatMsgType.OK)

    if failures:
        raise click.ClickException(
            f"{failures} retrieval configuration(s) failed."
        )

    return 0
