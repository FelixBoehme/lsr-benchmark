import gzip
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import milvus_retrieval


@pytest.fixture(scope="module")
def milvus_client(tmp_path_factory):
    server_path = tmp_path_factory.mktemp("milvus-server")
    with milvus_retrieval.milvus_server(server_path) as client:
        yield client


def test_embedding_conversion_merges_duplicate_indices_and_removes_zeros():
    vector = milvus_retrieval.to_sparse_vector(
        ["2", "0", "2", "1"],
        [0.25, 1.0, 0.75, 0.0],
    )

    assert vector == {0: 1.0, 2: 1.0}


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
        milvus_retrieval.to_sparse_vector(tokens, values)


def build_test_index(client, algorithm="DAAT_MAXSCORE"):
    return milvus_retrieval.build_index(
        client,
        [
            ("d1", ["0"], [1.0]),
            ("d2", ["0", "1"], [0.5, 0.25]),
            ("d3", ["1"], [1.0]),
            ("empty", ["2"], [0.0]),
        ],
        batch_size=2,
        algorithm=algorithm,
        drop_ratio_build=0.0,
    )


@pytest.mark.parametrize("algorithm", ["DAAT_MAXSCORE", "DAAT_WAND", "TAAT_NAIVE"])
def test_native_sparse_retrieval_returns_inner_product_top_k(
    milvus_client,
    algorithm,
):
    index = build_test_index(milvus_client, algorithm)

    results = list(
        milvus_retrieval.retrieve(
            milvus_client,
            index,
            [("q1", ["0"], [1.0])],
            k=2,
            batch_size=1,
            drop_ratio_search=0.0,
        )
    )

    assert results[0][0] == "q1"
    assert [document_id for document_id, _ in results[0][1]] == ["d1", "d2"]
    assert [score for _, score in results[0][1]] == pytest.approx([1.0, 0.5])


def test_retrieval_handles_multiple_queries_and_k_larger_than_corpus(milvus_client):
    index = build_test_index(milvus_client)

    results = list(
        milvus_retrieval.retrieve(
            milvus_client,
            index,
            [
                ("q1", ["0"], [1.0]),
                ("empty-query", [], []),
                ("q2", ["1"], [1.0]),
            ],
            k=10,
            batch_size=2,
            drop_ratio_search=0.0,
        )
    )

    assert [query_id for query_id, _ in results] == ["q1", "empty-query", "q2"]
    assert results[0][1][0][0] == "d1"
    assert results[1][1] == []
    assert results[2][1][0][0] == "d3"
    assert all(len(ranking) <= 3 for _, ranking in results)


def test_build_index_rejects_duplicate_document_ids(milvus_client):
    with pytest.raises(ValueError, match="must be unique"):
        milvus_retrieval.build_index(
            milvus_client,
            [
                ("duplicate", ["0"], [1.0]),
                ("duplicate", ["1"], [1.0]),
            ],
            batch_size=10,
            algorithm="DAAT_MAXSCORE",
            drop_ratio_build=0.0,
        )


def test_build_index_rejects_invalid_algorithm(milvus_client):
    with pytest.raises(ValueError, match="Unsupported"):
        milvus_retrieval.build_index(
            milvus_client,
            [("d1", ["0"], [1.0])],
            batch_size=10,
            algorithm="INVALID",
            drop_ratio_build=0.0,
        )


def test_retrieval_filters_invalid_and_non_positive_results():
    class FakeClient:
        def get_collection_stats(self, collection_name):
            return {"row_count": 3}

        def describe_index(self, collection_name, index_name):
            return {
                "metric_type": "IP",
                "inverted_index_algo": "DAAT_MAXSCORE",
            }

        def search(self, *args, **kwargs):
            return [
                [
                    {
                        "entity": {milvus_retrieval.DOCUMENT_ID_FIELD: "lower"},
                        "distance": 0.5,
                    },
                    {"entity": {}, "distance": 2.0},
                    {
                        "entity": {milvus_retrieval.DOCUMENT_ID_FIELD: "negative"},
                        "distance": -1.0,
                    },
                    {
                        "entity": {milvus_retrieval.DOCUMENT_ID_FIELD: "higher"},
                        "distance": 1.0,
                    },
                ]
            ]

    index = milvus_retrieval.MilvusIndex("test", 3, 3, "DAAT_MAXSCORE")
    results = list(
        milvus_retrieval.retrieve(
            FakeClient(),
            index,
            [("q1", ["0"], [1.0])],
            k=3,
            batch_size=1,
            drop_ratio_search=0.0,
        )
    )

    assert results == [("q1", [("higher", 1.0), ("lower", 0.5)])]


def test_retrieval_rejects_inconsistent_index_metadata():
    client = SimpleNamespace(
        get_collection_stats=lambda collection_name: {"row_count": 2}
    )
    index = milvus_retrieval.MilvusIndex("test", 3, 3, "DAAT_MAXSCORE")

    with pytest.raises(ValueError, match="metadata is inconsistent"):
        list(
            milvus_retrieval.retrieve(
                client,
                index,
                [("q1", ["0"], [1.0])],
                k=1,
                batch_size=1,
                drop_ratio_search=0.0,
            )
        )


def test_main_writes_a_compressed_trec_run(monkeypatch, tmp_path, milvus_client):
    embeddings = {
        "doc": [
            ("document-with-text-id", ["0"], [1.0]),
            ("d2", ["1"], [1.0]),
        ],
        "query": [("query-with-text-id", ["0"], [1.0])],
    }
    monkeypatch.setattr(
        milvus_retrieval.lsr_benchmark,
        "register_to_ir_datasets",
        lambda dataset: None,
    )
    monkeypatch.setattr(
        milvus_retrieval,
        "load_embeddings",
        lambda dataset, embedding, text_type: embeddings[text_type],
    )
    monkeypatch.setattr(milvus_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(milvus_retrieval, "tracking", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        milvus_retrieval,
        "milvus_server",
        lambda temporary_directory: nullcontext(milvus_client),
    )

    milvus_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="lightning-ir/naver-splade-v3-doc",
        output=tmp_path,
        k=10,
        algorithm="DAAT_MAXSCORE",
        index_batch_size=2,
        query_batch_size=2,
        drop_ratio_build=0.0,
        drop_ratio_search=0.0,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        assert (
            run_file.read().strip()
            == "query-with-text-id Q0 document-with-text-id 1 1.0 milvus"
        )
