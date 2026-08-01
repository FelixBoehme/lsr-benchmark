import click
from pathlib import Path
import pytest
from click.testing import CliRunner

from lsr_benchmark import main
from lsr_benchmark.datasets import (
    IR_DATASET_TO_TIRA_DATASET,
    all_dense_embeddings,
    all_embeddings,
)
from lsr_benchmark.retrieval_suites import RETRIEVAL_SUITES

from ._retrieval import ChoiceOrPath, resolve_retrieval_configuration

CHOICE_OR_PATH = ChoiceOrPath(
    [
        "all",
        "none",
    ]
    + all_embeddings()
    + list(all_dense_embeddings())
)


def test_return_valid_choice():
    assert CHOICE_OR_PATH.convert("bm25", None, None) == "bm25"


def test_fail_on_invalid_choice():
    with pytest.raises(click.BadParameter) as exec_info:
        CHOICE_OR_PATH.convert("bm255", None, None)
    assert "'bm255' is not one of" in str(exec_info.value)


def test_return_dir_path(tmp_path):
    assert CHOICE_OR_PATH.convert(str(tmp_path), None, None) == tmp_path


def test_fail_on_invalid_dir_path():
    with pytest.raises(click.BadParameter) as exec_info:
        CHOICE_OR_PATH.convert("/some/path", None, None)
    assert "'/some/path' is not one of" in str(exec_info.value)


def test_fail_on_file(tmp_path):
    file_path = tmp_path / "file.txt"
    with pytest.raises(click.BadParameter) as exec_info:
        CHOICE_OR_PATH.convert(str(file_path), None, None)
    assert f"'{str(file_path)}' is not one of" in str(exec_info.value)


def test_resolve_retrieval_suite():
    approaches, datasets, embeddings = resolve_retrieval_configuration(
        "reneuir-2026/small", (), (), ()
    )

    assert approaches == tuple(
        RETRIEVAL_SUITES["reneuir-2026/small"]["retrieval_engines"]
    )
    assert datasets == (
        IR_DATASET_TO_TIRA_DATASET["msmarco-passage/trec-dl-2019/judged"],
    )
    assert embeddings == ("naver-splade-v3",)


@pytest.mark.parametrize(
    ("approaches", "datasets", "embeddings"),
    [
        (("seismic",), (), ()),
        ((), ("tiny-example-20251002_0-training",), ()),
        ((), (), ("naver-splade-v3",)),
    ],
)
def test_suite_cannot_be_combined_with_manual_configuration(
    approaches, datasets, embeddings
):
    with pytest.raises(click.UsageError):
        resolve_retrieval_configuration(
            "reneuir-2026/full", approaches, datasets, embeddings
        )


def test_retrieval_accepts_suite(monkeypatch, tmp_path):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    calls = []

    monkeypatch.setattr(
        retrieval_module,
        "verify_docker_installation",
        lambda: (retrieval_module.FormatMsgType.OK, ""),
    )
    monkeypatch.setattr(
        retrieval_module, "docker_supported_target_platform", lambda: "linux/amd64"
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_approach_to_execution",
        lambda approaches, platform, embedding, print_message: {
            approach: {"tag": approach, "command": "run"} for approach in approaches
        },
    )
    monkeypatch.setattr(
        retrieval_module,
        "run_retrieval_engine",
        lambda image, command, dataset, embedding, output_dir, **kwargs: calls.append(
            (image, dataset, embedding, output_dir, kwargs)
        ),
    )
    monkeypatch.setattr(retrieval_module.os, "system", lambda command: 0)

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--suite",
            "reneuir-2026/small",
            "--out",
            str(tmp_path),
            "--cpus",
            "4",
            "--memory",
            "16g",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == (
        len(RETRIEVAL_SUITES["reneuir-2026/small"]["retrieval_engines"])
        * len(RETRIEVAL_SUITES["reneuir-2026/small"]["datasets"])
        * len(RETRIEVAL_SUITES["reneuir-2026/small"]["embeddings"])
    )
    assert {call[1] for call in calls} == {
        "trec-28-deep-learning-passages-20250926-training"
    }
    assert {call[4]["cpus"] for call in calls} == {4}
    assert {call[4]["memory"] for call in calls} == {"16g"}
    assert {call[4]["platform"] for call in calls} == {"linux/amd64"}


def test_run_retrieval_engine_forwards_resources_to_tira(monkeypatch, tmp_path):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    dataset_dir = tmp_path / "dataset"
    embedding_dir = tmp_path / "embeddings"
    execution_dir = tmp_path / "execution"
    dataset_dir.mkdir()
    embedding_dir.mkdir()
    execution_arguments = {}

    class LocalExecution:
        def run(self, output_dir, **kwargs):
            output_dir = Path(output_dir)
            (output_dir / "retrieval-metadata.yml").write_text("tag: test\n")
            kwargs["output_dir"] = output_dir
            execution_arguments.update(kwargs)

    class TiraClient:
        local_execution = LocalExecution()

    monkeypatch.setattr(retrieval_module, "Client", TiraClient)
    monkeypatch.setattr(
        retrieval_module.MonitoredExecution,
        "run",
        lambda self, method: (
            execution_dir.mkdir(),
            (execution_dir / "output").mkdir(),
            method(execution_dir / "output"),
            execution_dir,
        )[-1],
    )
    monkeypatch.setattr(
        retrieval_module,
        "check_format",
        lambda directory, expected_files, context: (
            retrieval_module.FormatMsgType.OK,
            "",
        ),
    )

    retrieval_module.run_retrieval_engine(
        "image",
        "command",
        dataset_dir,
        embedding_dir,
        cpus=8,
        memory="32g",
    )

    assert execution_arguments["cpu_count"] == 8
    assert execution_arguments["mem_limit"] == "32g"


def test_retrieval_reports_execution_failures(monkeypatch, tmp_path):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )

    monkeypatch.setattr(
        retrieval_module,
        "verify_docker_installation",
        lambda: (retrieval_module.FormatMsgType.OK, ""),
    )
    monkeypatch.setattr(
        retrieval_module, "docker_supported_target_platform", lambda: "linux/amd64"
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_approach_to_execution",
        lambda approaches, platform, embedding, print_message: {
            approach: {"tag": approach, "command": "run"} for approach in approaches
        },
    )
    monkeypatch.setattr(
        retrieval_module,
        "run_retrieval_engine",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("dataset lookup failed")
        ),
    )
    monkeypatch.setattr(retrieval_module.os, "system", lambda command: 0)

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--suite",
            "reneuir-2026/small",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "dataset lookup failed" in result.output
    assert "retrieval configuration(s) failed" in result.output
