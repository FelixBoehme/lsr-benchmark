from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lsr_benchmark._commands.download_embeddings import download_embeddings


@pytest.fixture
def mock_all_dense_embeddings():
    with patch("lsr_benchmark._commands.download_embeddings.all_dense_embeddings") as mock:
        mock.return_value = ["some-dense-model", "e5-mistral-7b-instruct"]
        yield mock


@pytest.fixture
def mock_patch_ir_metadata():
    with patch("lsr_benchmark._commands.download_embeddings.patch_ir_metadata") as mock:
        yield mock


@pytest.fixture
def mock_tira():
    tira = MagicMock()
    tira.get_run_output.side_effect = lambda run_id, dataset: Path(f"/mocked/{run_id}/{dataset}")
    return tira


def test_download_embeddings_with_path(mock_tira, mock_patch_ir_metadata):
    embedding_path = Path("/local/path/to/embeddings")

    result = download_embeddings(embedding_path, "my_dataset", mock_tira)

    assert result == embedding_path.resolve()
    mock_tira.get_run_output.assert_not_called()
    assert mock_patch_ir_metadata.call_count == 2


@pytest.mark.parametrize(
    "embedding, expected_run_id",
    [
        ("unknown-model", "lsr-benchmark/lightning-ir/unknown-model"),
        ("e5-mistral-7b-instruct", "lsr-benchmark/mteb/e5-mistral-7b-instruct"),
        ("some-dense-model", "lsr-benchmark/sentence-transformers/some-dense-model"),
    ],
)
def test_download_embeddings_routing(
    embedding, expected_run_id, mock_tira, mock_patch_ir_metadata, mock_all_dense_embeddings
):
    dataset = "my_dataset"

    result = download_embeddings(embedding, dataset, mock_tira)

    mock_tira.get_run_output.assert_called_once_with(expected_run_id, dataset)
    assert result == Path(f"/mocked/{expected_run_id}/{dataset}")


def test_download_embeddings_none_raises_error(mock_tira, mock_patch_ir_metadata):
    with pytest.raises(ValueError, match="Unable to download unknown embeddings 'None'"):
        download_embeddings("None", "my_dataset", mock_tira)

    mock_tira.get_run_output.assert_not_called()
    mock_patch_ir_metadata.assert_not_called()


def test_patch_ir_metadata_correctly_called(mock_tira, mock_patch_ir_metadata, mock_all_dense_embeddings):
    embedding = "some-dense-model"
    dataset = "my_dataset"

    result = download_embeddings(embedding, dataset, mock_tira)

    assert mock_patch_ir_metadata.call_count == 2

    mock_patch_ir_metadata.assert_any_call(
        result / "doc",
        {"data": {"test collection": {"name": "/tira-data/input"}}},
        {"data": {"test collection": {"name": dataset}}},
    )
    mock_patch_ir_metadata.assert_any_call(
        result / "query",
        {"data": {"test collection": {"name": "/tira-data/input"}}},
        {"data": {"test collection": {"name": dataset}}},
    )
