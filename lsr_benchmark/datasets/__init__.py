import json
from pathlib import Path

def lsr_overview():
    return json.loads((Path(__file__).parent / "overview.json").read_text())

def all_embeddings():
    overview = lsr_overview()
    ret = set()

    for dataset_id, stats in overview.items():
        for embedding, embedding_size in stats['embedding-sizes'].items():
            ret.add(embedding)
    return sorted(list(ret))

def all_dense_embeddings():
    return set([
        "all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2",
        "msmarco-MiniLM-L6-v3", "paraphrase-TinyBERT-L6-v2",
        "all-mpnet-base-v2", "intfloat-e5-large-v2",
        "BAAI-bge-m3", "BAAI-bge-large-en-v1.5",
        "e5-mistral-7b-instruct", "SFR-Embedding-Mistral",
        "Linq-Embed-Mistral", "Octen-Embedding-8B", "Qwen3-Embedding-8B",
        "speed-embedding-7b-instruct",
    ])

def all_datasets():
    overview = lsr_overview()
    ret = list(overview.keys())
    # TODO Integrate this better
    ret += ["aila-statutes-20260426-training", "aila-casedocs-20260426-training", "financebench-retrieval-20260427-training", "legal-summarization-20260427-training"]
    return sorted(ret)


TIRA_DATASET_ID_TO_IR_DATASET_ID = {
    'trec-18-web-20251008-test': 'clueweb09/en/trec-web-2009',
    'trec-19-web-20251008-test': 'clueweb09/en/trec-web-2010',
    'trec-20-web-20251008-test': 'clueweb09/en/trec-web-2011',
    'trec-21-web-20251008-test': 'clueweb09/en/trec-web-2012',
    'trec-22-web-20251008-test': 'clueweb12/trec-web-2013',
    'trec-23-web-20251008-test': 'clueweb12/trec-web-2014',
    'trec-28-deep-learning-passages-20250926-training': 'msmarco-passage/trec-dl-2019/judged',
    'trec-28-misinfo-20251008_1-test': 'clueweb12/b13/trec-misinfo-2019',
    'trec-29-deep-learning-passages-20250926-training': 'msmarco-passage/trec-dl-2020/judged',
    'trec-33-rag-20250926_1-training': 'msmarco-segment-v2.1/trec-rag-2024',
    'trec-robust-2004-fold-1-20250927-test': 'disks45/nocr/trec-robust-2004/fold1',
    'trec-robust-2004-fold-2-20250926-test': 'disks45/nocr/trec-robust-2004/fold2',
    'trec-robust-2004-fold-3-20250926-test': 'disks45/nocr/trec-robust-2004/fold3',
    'trec-robust-2004-fold-4-20250926-test': 'disks45/nocr/trec-robust-2004/fold4',
    'trec-robust-2004-fold-5-20250926-test': 'disks45/nocr/trec-robust-2004/fold5',
    'aila-statutes-20260426-training': 'rteb/aila/statutes',
    'aila-casedocs-20260426-training': 'rteb/aila/casedocs',
    'legal-summarization-20260427-training': 'rteb/legal-summarization',
    'financebench-retrieval-20260427-training': 'rteb/financebench',
}

EMBEDDING_MODEL_TO_PRETTY_NAME = {
    "naver-splade-v3": "Splade 3",
    "webis-splade": "Splade (webis)",
    "naver-splade-v3-distilbert": "Splade 3 Dist.",
    "naver-splade_v2_distil": "Splade 2 Dist.",
    "naver-splade-v3-doc": "Splade 3 Doc",
    "castorini-unicoil-noexp-msmarco-passage": "UniCoil",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v2-mini": "OS 2 Mini Doc",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v3-distill": "OS 3 Dist. Doc",
    "naver-splade-v3-lexical": "Splade 3 Lex.",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v2-distill": "OS 2 Dist. Doc",
    "opensearch-project-opensearch-neural-sparse-encoding-v2-distill": "OS 2 Dist.",
    "bge-m3": "BGE m3",
    "bm25": "BM25",
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L3-v2": "paraphrase-MiniLM-L3-v2",
    "msmarco-MiniLM-L6-v3": "msmarco-MiniLM-L6-v3",
}

EMBEDDING_MODEL_TO_HF_NAME = {
    "naver-splade-v3": "https://huggingface.co/naver/splade-v3",
    "webis-splade": "https://huggingface.co/webis/splade",
    "naver-splade-v3-distilbert": "https://huggingface.co/naver/splade-v3-distilbert",
    "naver-splade_v2_distil": "https://huggingface.co/naver/splade_v2_distil",
    "naver-splade-v3-doc": "https://huggingface.co/naver/splade-v3-doc",
    "castorini-unicoil-noexp-msmarco-passage": "https://huggingface.co/castorini/unicoil-noexp-msmarco-passage",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v2-mini": "https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v3-distill": "https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    "naver-splade-v3-lexical": "https://huggingface.co/naver/splade-v3-lexical",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v2-distill": "https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    "opensearch-project-opensearch-neural-sparse-encoding-v2-distill": "https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
    "bge-m3": "https://huggingface.co/BAAI/bge-m3",
    "bm25": "https://pages.nist.gov/trec-browser/trec4/proceedings/#okapi-at-trec-4",
}

EMBEDDING_MODEL_TO_ENGINE = {
    "naver-splade-v3": "lightning-ir",
    "webis-splade": "lightning-ir",
    "naver-splade-v3-distilbert": "lightning-ir",
    "naver-splade_v2_distil": "lightning-ir",
    "naver-splade-v3-doc": "lightning-ir",
    "castorini-unicoil-noexp-msmarco-passage": "lightning-ir",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v2-mini": "lightning-ir",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v3-distill": "lightning-ir",
    "naver-splade-v3-lexical": "lightning-ir",
    "opensearch-project-opensearch-neural-sparse-encoding-doc-v2-distill": "lightning-ir",
    "opensearch-project-opensearch-neural-sparse-encoding-v2-distill": "lightning-ir",
    "all-MiniLM-L6-v2": "sentence-transformers",
    "paraphrase-MiniLM-L3-v2": "sentence-transformers",
    "msmarco-MiniLM-L6-v3": "sentence-transformers",
    "bge-m3": "FlagEmbedding",
    "bm25": "PyTerrier"
}

EMBEDDING_ENGINE_LINKS = {
    "PyTerrier": "https://github.com/reneuir/lsr-benchmark/blob/main/step-02-embedding-approaches/lexical/run-pyterrier.py",
    "FlagEmbedding": "https://github.com/reneuir/lsr-benchmark/blob/main/step-02-embedding-approaches/bge-m3/bgem3.py",
    "lightning-ir": "https://github.com/reneuir/lsr-benchmark/blob/main/step-02-embedding-approaches/lightning-ir/lightning-ir.py",
}

IR_DATASET_TO_TIRA_DATASET = {v:k for k, v in TIRA_DATASET_ID_TO_IR_DATASET_ID.items()}

SUPPORTED_IR_DATASETS = sorted(list(TIRA_DATASET_ID_TO_IR_DATASET_ID.keys()) + ["tiny-example-20251002_0-training"])

def all_ir_datasets():
    return sorted([TIRA_DATASET_ID_TO_IR_DATASET_ID[i] for i in all_datasets() if i in TIRA_DATASET_ID_TO_IR_DATASET_ID and TIRA_DATASET_ID_TO_IR_DATASET_ID[i]])
