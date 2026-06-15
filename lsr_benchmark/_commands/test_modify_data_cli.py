import io
import json

import numpy as np
import pytest

from lsr_benchmark.datasets import all_datasets

from ._modify_data import DATASET_TO_MAPPING, load_and_merge_embeddings, prefix_json, quantize


def test_mapping_for_all_datasets():
    unmapped = [d for d in all_datasets() if d not in DATASET_TO_MAPPING]
    assert not unmapped, f"Mappings missing for: {unmapped}"


def test_quantize_1_2_4_bits():
    arr = np.array([0.0, 0.5, 1.0])

    res1 = quantize(arr, 1)
    assert np.array_equal(res1, np.array([0, 0, 1], dtype=np.int8))

    res2 = quantize(arr, 2)
    assert np.array_equal(res2, np.array([0, 2, 3], dtype=np.int8))

    res4 = quantize(arr, 4)
    assert np.array_equal(res4, np.array([0, 8, 15], dtype=np.int8))


def test_quantize_8_bits():
    arr = np.array([0.0, 0.5, 1.0])
    res = quantize(arr, 8)
    assert np.array_equal(res, np.array([0, 127, 255], dtype=np.uint8))


def test_quantize_16_bits():
    arr = np.array([1.234, 5.678], dtype=np.float32)
    res = quantize(arr, 16)
    assert res.dtype == np.float16
    assert np.array_equal(res, np.array([1.234, 5.678], dtype=np.float16))


def test_quantize_unsupported_bits():
    with pytest.raises(ValueError, match="Quantizing to 3 bits is not supported."):
        quantize(np.array([1.0]), 3)


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
