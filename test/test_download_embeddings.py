from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lsr_benchmark._commands.download_embeddings import download_embeddings

META_TEMPLATE = {"data": {"test collection": {"name": "/tira-data/input"}}}


def write_meta_files(base_dir: Path):
    for folder in ["doc", "query"]:
        folder_dir = base_dir / folder
        folder_dir.mkdir(parents=True, exist_ok=True)
        with open(folder_dir / f"{folder}-ir-metadata.yml", "w") as f:
            yaml.dump(META_TEMPLATE, f)


def read_meta_files(base_dir: Path):
    metas = {}
    for folder in ["doc", "query"]:
        with open(base_dir / folder / f"{folder}-ir-metadata.yml", "r") as f:
            metas[folder] = yaml.safe_load(f)
    return metas


@pytest.fixture
def mock_all_dense_embeddings():
    with patch("lsr_benchmark._commands.download_embeddings.all_dense_embeddings") as mock:
        mock.return_value = ["some-dense-model", "e5-mistral-7b-instruct"]
        yield mock


@pytest.fixture
def mock_tira(tmp_path):
    tira = MagicMock()

    def get_run_output(run_id, dataset):
        out_dir = tmp_path / run_id / dataset
        write_meta_files(out_dir)
        return out_dir

    tira.get_run_output.side_effect = get_run_output
    return tira


def test_download_embeddings_with_path(tmp_path, mock_tira):
    embedding_path = tmp_path / "local" / "path" / "to" / "embeddings"
    write_meta_files(embedding_path)

    result = download_embeddings(embedding_path, "my_dataset", mock_tira)

    assert result == embedding_path.resolve()
    mock_tira.get_run_output.assert_not_called()

    for meta in read_meta_files(result).values():
        assert meta["data"]["test collection"]["name"] == "my_dataset"


@pytest.mark.parametrize(
    "embedding, expected_run_id",
    [
        ("unknown-model", "lsr-benchmark/lightning-ir/unknown-model"),
        ("e5-mistral-7b-instruct", "lsr-benchmark/mteb/e5-mistral-7b-instruct"),
        ("some-dense-model", "lsr-benchmark/sentence-transformers/some-dense-model"),
    ],
)
def test_download_embeddings_routing(embedding, expected_run_id, tmp_path, mock_tira, mock_all_dense_embeddings):
    dataset = "my_dataset"

    result = download_embeddings(embedding, dataset, mock_tira)

    mock_tira.get_run_output.assert_called_once_with(expected_run_id, dataset)
    assert result == tmp_path / expected_run_id / dataset


def test_download_embeddings_none_raises_error(mock_tira):
    with pytest.raises(ValueError, match="Unable to download unknown embeddings 'None'"):
        download_embeddings("None", "my_dataset", mock_tira)

    mock_tira.get_run_output.assert_not_called()


def test_metadata_updated_with_dataset_and_embedding_model(mock_tira, mock_all_dense_embeddings):
    embedding = "some-dense-model"
    dataset = "my_dataset"

    result = download_embeddings(embedding, dataset, mock_tira)

    for meta in read_meta_files(result).values():
        assert meta["data"]["test collection"]["name"] == dataset
        assert meta["data"]["embedding model"] == {"name": embedding}
