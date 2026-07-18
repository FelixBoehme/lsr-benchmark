RETRIEVAL_SUITES = {
    "reneuir-2026/small": {
        "datasets": ["msmarco-passage/trec-dl-2019/judged"],
        "embeddings": ["naver-splade-v3"],
        "retrieval_engines": [
            "duckdb",
            "kannolo",
            "naive-search",
            "pyterrier-splade",
            "pyterrier-splade-pisa",
            "seismic",
        ],
    },
    "reneuir-2026/full": {
        "datasets": ["msmarco-passage/trec-dl-2019/judged", "clueweb09/en/trec-web-2009", "disks45/nocr/trec-robust-2004/fold1"],
        "embeddings": ["naver-splade-v3", "webis-splade", "opensearch-project-opensearch-neural-sparse-encoding-v2-distill", "opensearch-project-opensearch-neural-sparse-encoding-doc-v3-distill"],
        "retrieval_engines": [
            "duckdb",
            "kannolo",
            "naive-search",
            "pyterrier-splade",
            "pyterrier-splade-pisa",
            "seismic",
        ],
    },
}
