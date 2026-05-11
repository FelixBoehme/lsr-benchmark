import click
import pytest

from lsr_benchmark.datasets import all_dense_embeddings, all_embeddings

from ._retrieval import ChoiceOrPath

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
