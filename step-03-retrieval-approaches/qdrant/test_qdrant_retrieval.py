import gzip
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import qdrant_retrieval


@pytest.fixture(scope="module")
def qdrant_client(tmp_path_factory):
    server_path = tmp_path_factory.mktemp("qdrant-server")
    with qdrant_retrieval.qdrant_server(server_path) as client:
        yield client


def test_embedding_conversion_merges_duplicate_indices_and_removes_zeros():
    vector = qdrant_retrieval.to_sparse_vector(
        ["2", "0", "2", "1"],
        [0.25, 1.0, 0.75, 0.0],
    )

    assert vector.indices == [0, 2]
    assert vector.values == [1.0, 1.0]


@pytest.mark.parametrize(
    ("tokens", "values", "message"),
    [
        (["0"], [], "same length"),
        (["not-an-index"], [1.0], "not an integer"),
        (["-1"], [1.0], "unsigned 32-bit"),
        (["0"], [float("inf")], "finite"),
    ],
)
def test_embedding_conversion_rejects_invalid_vectors(tokens, values, message):
    with pytest.raises(ValueError, match=message):
        qdrant_retrieval.to_sparse_vector(tokens, values)


def build_test_index(client):
    return qdrant_retrieval.build_index(
        client,
        [
            ("d1", ["0"], [1.0]),
            ("d2", ["0", "1"], [0.5, 0.25]),
            ("d3", ["1"], [1.0]),
            ("empty", ["2"], [0.0]),
        ],
        batch_size=2,
        on_disk=False,
    )


def test_native_sparse_retrieval_returns_inner_product_top_k(qdrant_client):
    index = build_test_index(qdrant_client)

    results = list(
        qdrant_retrieval.retrieve(
            qdrant_client,
            index,
            [("q1", ["0"], [1.0])],
            k=2,
            batch_size=1,
        )
    )

    assert results[0][0] == "q1"
    assert [document_id for document_id, _ in results[0][1]] == ["d1", "d2"]
    assert [score for _, score in results[0][1]] == pytest.approx([1.0, 0.5])


def test_retrieval_handles_multiple_queries_and_k_larger_than_corpus(qdrant_client):
    index = build_test_index(qdrant_client)

    results = list(
        qdrant_retrieval.retrieve(
            qdrant_client,
            index,
            [
                ("q1", ["0"], [1.0]),
                ("empty-query", [], []),
                ("q2", ["1"], [1.0]),
            ],
            k=10,
            batch_size=2,
        )
    )

    assert [query_id for query_id, _ in results] == ["q1", "empty-query", "q2"]
    assert results[0][1][0][0] == "d1"
    assert results[1][1] == []
    assert results[2][1][0][0] == "d3"
    assert all(len(ranking) <= 3 for _, ranking in results)


def test_build_index_rejects_duplicate_document_ids(qdrant_client):
    with pytest.raises(ValueError, match="must be unique"):
        qdrant_retrieval.build_index(
            qdrant_client,
            [
                ("duplicate", ["0"], [1.0]),
                ("duplicate", ["1"], [1.0]),
            ],
            batch_size=10,
            on_disk=False,
        )


def test_retrieval_filters_invalid_and_non_positive_results():
    class FakeClient:
        def get_collection(self, collection_name):
            return SimpleNamespace(points_count=3)

        def query_batch_points(self, collection_name, requests):
            return [
                SimpleNamespace(
                    points=[
                        SimpleNamespace(payload={"doc_id": "lower"}, score=0.5),
                        SimpleNamespace(payload=None, score=2.0),
                        SimpleNamespace(payload={"doc_id": "negative"}, score=-1.0),
                        SimpleNamespace(payload={"doc_id": "higher"}, score=1.0),
                    ]
                )
            ]

    index = qdrant_retrieval.QdrantIndex("test", 3, 3)
    results = list(
        qdrant_retrieval.retrieve(
            FakeClient(),
            index,
            [("q1", ["0"], [1.0])],
            k=3,
            batch_size=1,
        )
    )

    assert results == [("q1", [("higher", 1.0), ("lower", 0.5)])]


def test_retrieval_rejects_inconsistent_index_metadata():
    client = SimpleNamespace(
        get_collection=lambda collection_name: SimpleNamespace(points_count=2)
    )
    index = qdrant_retrieval.QdrantIndex("test", 3, 3)

    with pytest.raises(ValueError, match="metadata is inconsistent"):
        list(
            qdrant_retrieval.retrieve(
                client,
                index,
                [("q1", ["0"], [1.0])],
                k=1,
                batch_size=1,
            )
        )


def test_main_writes_a_compressed_trec_run(monkeypatch, tmp_path, qdrant_client):
    embeddings = {
        "doc": [
            ("document-with-text-id", ["0"], [1.0]),
            ("d2", ["1"], [1.0]),
        ],
        "query": [("query-with-text-id", ["0"], [1.0])],
    }
    monkeypatch.setattr(
        qdrant_retrieval.lsr_benchmark,
        "register_to_ir_datasets",
        lambda dataset: None,
    )
    monkeypatch.setattr(
        qdrant_retrieval,
        "load_embeddings",
        lambda dataset, embedding, text_type: embeddings[text_type],
    )
    monkeypatch.setattr(qdrant_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(qdrant_retrieval, "tracking", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        qdrant_retrieval,
        "qdrant_server",
        lambda temporary_directory: nullcontext(qdrant_client),
    )

    qdrant_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="lightning-ir/naver-splade-v3-doc",
        output=tmp_path,
        k=10,
        index_batch_size=2,
        query_batch_size=2,
        on_disk=False,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        assert (
            run_file.read().strip()
            == "query-with-text-id Q0 document-with-text-id 1 1.0 qdrant"
        )
