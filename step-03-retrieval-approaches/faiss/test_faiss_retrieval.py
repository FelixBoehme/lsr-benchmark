import gzip
from contextlib import nullcontext

import numpy as np
import pytest

import faiss_retrieval
from faiss_retrieval import (
    build_index,
    determine_dimension,
    retrieve,
    to_dense_matrix,
)


def test_sparse_embeddings_are_converted_to_dense_float32_matrix():
    embeddings = [
        ("d1", ["0", "2"], [1.0, 0.5]),
        ("d2", ["1"], [0.25]),
    ]

    ids, matrix = to_dense_matrix(embeddings, dimension=3)

    assert ids == ["d1", "d2"]
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(
        matrix,
        np.array([[1.0, 0.0, 0.5], [0.0, 0.25, 0.0]], dtype=np.float32),
    )


def test_dimension_covers_document_and_query_components():
    documents = [("d1", ["1"], [1.0])]
    queries = [("q1", ["4"], [1.0])]

    assert determine_dimension(documents, queries) == 5


def test_retrieve_returns_inner_product_top_k_in_score_order():
    doc_ids = ["d1", "d2", "d3"]
    documents = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    index = build_index(documents, "IP", None, None, None, None)
    results = retrieve(index, ["q1"], queries, doc_ids, k=2, index_type="IP", binary=False)

    assert [result[2] for result in results[0]] == ["d1", "d2"]
    assert [result[1] for result in results[0]] == pytest.approx([1.0, 0.8])


def test_retrieve_handles_multiple_queries_and_k_larger_than_corpus():
    doc_ids = ["d1", "d2"]
    documents = np.eye(2, dtype=np.float32)
    queries = np.eye(2, dtype=np.float32)

    index = build_index(documents, "IP", None, None, None, None)
    results = retrieve(index, ["q1", "q2"], queries, doc_ids, k=10, index_type="IP", binary=False)

    assert [[result[2] for result in ranking] for ranking in results] == [["d1"], ["d2"]]


def test_retrieve_omits_non_positive_scores_for_flat_ip():
    documents = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    index = build_index(documents, "IP", None, None, None, None)
    results = retrieve(index, ["q1"], queries, ["d1", "d2"], k=2, index_type="IP", binary=False)

    assert results == [[]]


def test_retrieve_rejects_non_positive_k():
    documents = np.array([[1.0]], dtype=np.float32)

    index = build_index(documents, "IP", None, None, None, None)
    with pytest.raises(ValueError, match="k must be at least 1"):
        retrieve(index, ["q1"], documents, ["d1"], k=0, index_type="IP", binary=False)


def test_build_index_rejects_embeddings_with_no_dimensions():
    with pytest.raises(ValueError, match="at least one dimension"):
        build_index(np.zeros((2, 0), dtype=np.float32), "IP", None, None, None, False)


def test_build_index_rejects_embeddings_that_are_not_two_dimensional():
    with pytest.raises(ValueError, match="at least one dimension"):
        build_index(np.zeros((2, 2, 2), dtype=np.float32), "IP", None, None, None, False)


def test_build_index_rejects_unsupported_index_type():
    documents = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="UNKNOWN"):
        build_index(documents, "UNKNOWN", None, None, None, False)


def test_build_index_hnsw_orders_neighbors_by_ascending_l2_distance():
    doc_ids = ["d1", "d2", "d3"]
    documents = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )
    queries = np.array([[0.9, 0.1]], dtype=np.float32)

    index = build_index(documents, "HNSW", 16, None, None, False)
    results = retrieve(index, ["q1"], queries, doc_ids, k=2, index_type="HNSW", binary=False)

    assert [result[2] for result in results[0]] == ["d2", "d1"]
    assert [result[1] for result in results[0]] == pytest.approx([0.02, 0.02], abs=1e-3)


def test_build_index_hnsw_keeps_exact_matches_with_zero_l2_distance():
    doc_ids = ["d1", "d2", "d3"]
    documents = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    index = build_index(documents, "HNSW", 16, None, None, False)
    results = retrieve(index, ["q1"], queries, doc_ids, k=2, index_type="HNSW", binary=False)

    assert [result[2] for result in results[0]] == ["d1", "d2"]
    assert [result[1] for result in results[0]] == pytest.approx([0.0, 0.08], abs=1e-3)


def test_build_index_ivf_retrieves_by_inner_product_and_respects_nprobe():
    doc_ids = ["d1", "d2", "d3", "d4"]
    documents = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    index = build_index(documents, "IVF", None, nlists=2, nprobe=2, binary=False)
    results = retrieve(index, ["q1"], queries, doc_ids, k=2, index_type="IVF", binary=False)

    assert index.nprobe == 2
    assert [result[2] for result in results[0]] == ["d1", "d2"]
    assert [result[1] for result in results[0]] == pytest.approx([1.0, 0.9], abs=1e-3)


def test_build_index_ivf_scores_are_raw_inner_products_not_cosine_normalized():
    doc_ids = ["d1", "d2"]
    documents = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    index = build_index(documents, "IVF", None, nlists=1, nprobe=1, binary=False)
    results = retrieve(index, ["q1"], queries, doc_ids, k=2, index_type="IVF", binary=False)

    assert [result[2] for result in results[0]] == ["d1"]
    assert results[0][0][1] == pytest.approx(2.0)


def test_retrieve_omits_non_positive_scores_for_ivf():
    doc_ids = ["d1", "d2"]
    documents = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    index = build_index(documents, "IVF", None, nlists=1, nprobe=1, binary=False)
    results = retrieve(index, ["q1"], queries, doc_ids, k=2, index_type="IVF", binary=False)

    assert results == [[]]


def test_build_index_and_retrieve_support_binary_vectors_with_ip():
    doc_ids = ["d1", "d2", "d3"]
    documents = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    queries = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.float32)

    index = build_index(documents, "IP", None, None, None, True)
    results = retrieve(index, ["q1"], queries, doc_ids, k=3, index_type="IP", binary=True)

    assert [result[2] for result in results[0]] == ["d1", "d2", "d3"]
    assert [result[1] for result in results[0]] == pytest.approx([0.0, 1.0, 8.0])


def test_build_index_and_retrieve_support_binary_vectors_with_hnsw():
    doc_ids = ["d1", "d2", "d3"]
    documents = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    queries = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.float32)

    index = build_index(documents, "HNSW", 16, None, None, True)
    results = retrieve(index, ["q1"], queries, doc_ids, k=3, index_type="HNSW", binary=True)

    assert [result[2] for result in results[0]] == ["d1", "d2", "d3"]
    assert [result[1] for result in results[0]] == pytest.approx([0.0, 1.0, 8.0])


def test_build_index_and_retrieve_support_binary_vectors_with_ivf():
    doc_ids = ["d1", "d2", "d3"]
    documents = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    queries = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.float32)

    index = build_index(documents, "IVF", None, nlists=1, nprobe=1, binary=True)
    results = retrieve(index, ["q1"], queries, doc_ids, k=3, index_type="IVF", binary=True)

    assert index.nprobe == 1
    assert [result[2] for result in results[0]] == ["d1", "d2", "d3"]
    assert [result[1] for result in results[0]] == pytest.approx([0.0, 1.0, 8.0])


def test_main_writes_a_trec_run(monkeypatch, tmp_path):
    embeddings = {
        "doc": [
            ("d1", ["0"], [1.0]),
            ("d2", ["1"], [1.0]),
        ],
        "query": [("q1", ["0"], [1.0])],
    }
    monkeypatch.setattr(faiss_retrieval.lsr_benchmark, "register_to_ir_datasets", lambda dataset: None)
    monkeypatch.setattr(
        faiss_retrieval,
        "load_embeddings",
        lambda dataset, embedding, text_type: embeddings[text_type],
    )
    monkeypatch.setattr(faiss_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(faiss_retrieval, "tracking", lambda **kwargs: nullcontext())

    faiss_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="lightning-ir/naver-splade-v3-doc",
        output=tmp_path,
        k=10,
        batch_size=128,
        index_type="IP",
        m=None,
        nlists=None,
        nprobe=None,
        binary=False,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        assert run_file.read().strip() == "q1 Q0 d1 1 1.0 faiss"
