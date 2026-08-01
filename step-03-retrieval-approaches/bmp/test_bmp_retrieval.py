import gzip
from contextlib import nullcontext

import pytest

import bmp_retrieval


def build_test_index(tmp_path):
    documents = [
        ("d1", ["0"], [1.0]),
        ("d2", ["0", "1"], [0.5, 0.25]),
        ("d3", ["1"], [1.0]),
    ]
    return bmp_retrieval.build_index(
        tmp_path / "index.bmp",
        documents,
        block_size=1,
        compress_range=True,
        max_document_impact=255,
    )


def test_merge_embedding_combines_duplicate_tokens_and_ignores_non_positive_values():
    embedding = bmp_retrieval.merge_embedding(
        ["0", "0", "1", "2"],
        [0.25, 0.75, 0.0, -1.0],
    )

    assert embedding == {"0": 1.0}


def test_quantize_embedding_preserves_relative_document_weights():
    quantized = bmp_retrieval.quantize_embedding(
        {"0": 1.0, "1": 0.5},
        maximum=100,
        reference_max=2.0,
    )

    assert quantized == {"0": 50, "1": 25}


def test_document_impact_is_capped_to_prevent_score_overflow():
    queries = [("q1", [str(token) for token in range(9)], [1.0] * 9)]

    maximum = bmp_retrieval.determine_max_document_impact(
        queries,
        requested_maximum=255,
    )

    assert maximum == 227
    assert maximum * bmp_retrieval.MAX_QUERY_WEIGHT * 9 <= bmp_retrieval.MAX_SCORE


def test_safe_document_impacts_preserve_multi_term_ranking(tmp_path):
    tokens = [str(token) for token in range(9)]
    queries = [("q1", tokens, [1.0] * len(tokens))]
    maximum = bmp_retrieval.determine_max_document_impact(queries, 255)
    index = bmp_retrieval.build_index(
        tmp_path / "index.bmp",
        [
            ("multi-term", tokens, [1.0] * len(tokens)),
            ("single-term", ["0"], [1.0]),
        ],
        block_size=1,
        compress_range=True,
        max_document_impact=maximum,
    )

    results = bmp_retrieval.retrieve(index, queries, 2, alpha=1.0, beta=1.0)

    assert [doc_id for doc_id, _ in results[0][1]] == [
        "multi-term",
        "single-term",
    ]


def test_retrieve_returns_top_k_in_descending_score_order(tmp_path):
    index = build_test_index(tmp_path)

    results = bmp_retrieval.retrieve(
        index,
        [("q1", ["0"], [1.0])],
        k=2,
        alpha=1.0,
        beta=1.0,
    )

    assert results[0][0] == "q1"
    assert [doc_id for doc_id, _ in results[0][1]] == ["d1", "d2"]
    assert results[0][1][0][1] > results[0][1][1][1]


def test_retrieve_fills_top_k_when_scores_are_tied(tmp_path):
    documents = [
        (f"d{doc_id}", ["0"], [1.0])
        for doc_id in range(10)
    ]
    index = bmp_retrieval.build_index(
        tmp_path / "index.bmp",
        documents,
        block_size=8,
        compress_range=True,
        max_document_impact=255,
    )

    results = bmp_retrieval.retrieve(
        index,
        [("q1", ["0"], [1.0])],
        k=10,
        alpha=1.0,
        beta=1.0,
    )

    assert len(results[0][1]) == 10


def test_retrieve_handles_multiple_queries_and_k_larger_than_corpus(tmp_path):
    index = build_test_index(tmp_path)

    results = bmp_retrieval.retrieve(
        index,
        [
            ("q1", ["0"], [1.0]),
            ("q2", ["1"], [1.0]),
        ],
        k=10,
        alpha=1.0,
        beta=1.0,
    )

    assert [query_id for query_id, _ in results] == ["q1", "q2"]
    assert results[0][1][0][0] == "d1"
    assert results[1][1][0][0] == "d3"
    assert all(len(ranking) <= 3 for _, ranking in results)


def test_retrieve_skips_empty_and_out_of_vocabulary_queries(tmp_path):
    index = build_test_index(tmp_path)

    results = bmp_retrieval.retrieve(
        index,
        [
            ("empty", [], []),
            ("unknown", ["999"], [1.0]),
            ("non-positive", ["0"], [0.0]),
        ],
        k=10,
        alpha=1.0,
        beta=1.0,
    )

    assert results == [
        ("empty", []),
        ("unknown", []),
        ("non-positive", []),
    ]


def test_build_index_rejects_inconsistent_embeddings(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        bmp_retrieval.build_index(
            tmp_path / "index.bmp",
            [("d1", ["0"], [])],
            block_size=8,
            compress_range=True,
            max_document_impact=255,
        )


def test_build_index_rejects_block_size_larger_than_u8_offset(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 256"):
        bmp_retrieval.build_index(
            tmp_path / "index.bmp",
            [("d1", ["0"], [1.0])],
            block_size=257,
            compress_range=True,
            max_document_impact=255,
        )


def test_build_index_rejects_vocabulary_larger_than_u16_term_ids(tmp_path):
    tokens = [str(token) for token in range(bmp_retrieval.MAX_VOCABULARY_SIZE + 1)]

    with pytest.raises(ValueError, match="65,536"):
        bmp_retrieval.build_index(
            tmp_path / "index.bmp",
            [("d1", tokens, [1.0] * len(tokens))],
            block_size=8,
            compress_range=True,
            max_document_impact=255,
        )


def test_retrieve_rejects_invalid_beta(tmp_path):
    index = build_test_index(tmp_path)

    with pytest.raises(ValueError, match="beta"):
        bmp_retrieval.retrieve(
            index,
            [("q1", ["0"], [1.0])],
            k=10,
            alpha=1.0,
            beta=0.0,
        )


def test_main_writes_a_compressed_trec_run(monkeypatch, tmp_path):
    embeddings = {
        "doc": [
            ("d1", ["0"], [1.0]),
            ("d2", ["0"], [0.5]),
            ("d3", ["1"], [1.0]),
        ],
        "query": [("query-with-text-id", ["0"], [1.0])],
    }
    monkeypatch.setattr(
        bmp_retrieval.lsr_benchmark,
        "register_to_ir_datasets",
        lambda dataset: None,
    )
    monkeypatch.setattr(
        bmp_retrieval,
        "load_embeddings",
        lambda dataset, embedding, text_type: embeddings[text_type],
    )
    monkeypatch.setattr(bmp_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(bmp_retrieval, "tracking", lambda **kwargs: nullcontext())

    bmp_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="lightning-ir/naver-splade-v3-doc",
        output=tmp_path,
        k=10,
        block_size=1,
        compress_range=True,
        max_document_impact=255,
        alpha=1.0,
        beta=1.0,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        lines = run_file.read().strip().splitlines()

    assert lines[0].split()[:4] == ["query-with-text-id", "Q0", "d1", "1"]
    assert lines[0].endswith(" bmp")
    assert all(" d3 " not in line for line in lines)
