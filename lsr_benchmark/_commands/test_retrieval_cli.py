import click
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
        "run_foo",
        lambda image, command, dataset, embedding, output_dir, platform: calls.append(
            (image, dataset, embedding, output_dir, platform)
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

    assert result.exit_code == 0, result.output
    assert len(calls) == (
        len(RETRIEVAL_SUITES["reneuir-2026/small"]["retrieval_engines"])
        * len(RETRIEVAL_SUITES["reneuir-2026/small"]["datasets"])
        * len(RETRIEVAL_SUITES["reneuir-2026/small"]["embeddings"])
    )
    assert {call[1] for call in calls} == {
        "trec-28-deep-learning-passages-20250926-training"
    }


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
        "run_foo",
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
