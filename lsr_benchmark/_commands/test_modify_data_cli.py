import io
import json

import numpy as np
import pytest

from ._modify_data import DuplicateBehaviour, load_and_merge_embeddings, prefix_json


def test_load_and_merge_embeddings(tmp_path):
    path1 = tmp_path / "run1"
    path1.mkdir()
    np.savez_compressed(
        path1 / "test.npz", data=np.array([0.1, 0.2]), indices=np.array([0, 1]), indptr=np.array([0, 2])
    )
    mask1 = np.array([True])

    path2 = tmp_path / "run2"
    path2.mkdir()
    np.savez_compressed(
        path2 / "test.npz", data=np.array([0.3, 0.4, 0.5]), indices=np.array([0, 1, 2]), indptr=np.array([0, 3])
    )
    mask2 = np.array([True])

    merged = load_and_merge_embeddings([path1, path2], "test.npz", keep_masks=[mask1, mask2])

    assert np.array_equal(merged["data"], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert np.array_equal(merged["indices"], np.array([0, 1, 0, 1, 2]))
    assert np.array_equal(merged["indptr"], np.array([0, 2, 5]))


def test_load_and_merge_embeddings_with_skips(tmp_path):
    path1 = tmp_path / "run1"
    path1.mkdir()
    np.savez_compressed(
        path1 / "test.npz",
        data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        indices=np.array([0, 0, 1, 0, 1, 2]),
        indptr=np.array([0, 1, 3, 6]),
    )

    mask1 = np.array([True, False, True])

    merged = load_and_merge_embeddings([path1], "test.npz", keep_masks=[mask1])

    assert np.array_equal(merged["data"], np.array([0.1, 0.4, 0.5, 0.6]))
    assert np.array_equal(merged["indices"], np.array([0, 0, 1, 2]))
    assert np.array_equal(merged["indptr"], np.array([0, 1, 4]))


def test_prefix_json_prefix_behaviour():
    input_data = '{"qid": "1", "text": "hello"}\n\n{"qid": "2", "text": "world"}\n'
    infile = io.StringIO(input_data)
    outfile = io.StringIO()
    seen_ids = set()

    prefix_json(infile, outfile, DuplicateBehaviour.PREFIX, "qid", seen_ids, "d1")

    outfile.seek(0)
    lines = outfile.readlines()

    assert len(lines) == 2  # empty line should be ignored
    assert json.loads(lines[0]) == {"qid": "d1-1", "text": "hello"}
    assert json.loads(lines[1]) == {"qid": "d1-2", "text": "world"}


def test_prefix_json_skip_behaviour():
    input_data = '{"qid": "1", "text": "hello"}\n{"qid": "1", "text": "hello"}\n{"qid": "2", "text": "world"}\n'
    infile = io.StringIO(input_data)
    outfile = io.StringIO()
    seen_ids = set()

    prefix_json(infile, outfile, DuplicateBehaviour.SKIP, "qid", seen_ids, "d1")

    outfile.seek(0)
    lines = outfile.readlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"qid": "1", "text": "hello"}
    assert json.loads(lines[1]) == {"qid": "2", "text": "world"}
    assert seen_ids == {"1", "2"}


def test_prefix_json_fail_behaviour():
    input_data = '{"qid": "1", "text": "hello"}\n{"qid": "1", "text": "hello"}\n'
    infile = io.StringIO(input_data)
    outfile = io.StringIO()
    seen_ids = set()

    with pytest.raises(ValueError, match="Duplicate qid '1' found while joining datasets."):
        prefix_json(infile, outfile, DuplicateBehaviour.FAIL, "qid", seen_ids, "d1")
