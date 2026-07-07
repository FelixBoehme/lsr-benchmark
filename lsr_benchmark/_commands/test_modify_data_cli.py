import io
import json

import numpy as np

from ._modify_data import load_and_merge_embeddings, prefix_json


def test_load_and_merge_embeddings(tmp_path):
    path1 = tmp_path / "run1"
    path1.mkdir()
    np.savez_compressed(
        path1 / "test.npz", data=np.array([0.1, 0.2]), indices=np.array([0, 1]), indptr=np.array([0, 2])
    )

    path2 = tmp_path / "run2"
    path2.mkdir()
    np.savez_compressed(
        path2 / "test.npz", data=np.array([0.3, 0.4, 0.5]), indices=np.array([0, 1, 2]), indptr=np.array([0, 3])
    )

    merged = load_and_merge_embeddings([path1, path2], "test.npz")

    assert np.array_equal(merged["data"], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert np.array_equal(merged["indices"], np.array([0, 1, 0, 1, 2]))
    assert np.array_equal(merged["indptr"], np.array([0, 2, 5]))


def test_prefix_json():
    input_data = '{"qid": "1", "text": "hello"}\n\n{"qid": "2", "text": "world"}\n'
    infile = io.StringIO(input_data)
    outfile = io.StringIO()

    prefix_json(infile, outfile, "d1", "qid")

    outfile.seek(0)
    lines = outfile.readlines()

    assert len(lines) == 2  # empty line should be ignored
    assert json.loads(lines[0]) == {"qid": "d1-1", "text": "hello"}
    assert json.loads(lines[1]) == {"qid": "d1-2", "text": "world"}
