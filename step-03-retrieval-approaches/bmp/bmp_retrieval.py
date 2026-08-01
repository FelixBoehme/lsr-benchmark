#!/usr/bin/env python3
import gzip
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from bmp import Indexer, Searcher
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings


MAX_SCORE = (1 << 16) - 1
MAX_QUERY_WEIGHT = 32
MAX_VOCABULARY_SIZE = 1 << 16


@dataclass(frozen=True)
class BmpIndex:
    searcher: Searcher
    vocabulary: frozenset[str]
    document_ids: frozenset[str]


def merge_embedding(tokens, values):
    if len(tokens) != len(values):
        raise ValueError("Embedding tokens and values must have the same length.")

    merged = defaultdict(float)
    for token, value in zip(tokens, values):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Embedding values must be finite.")
        if value > 0:
            merged[str(token)] += value
    return dict(merged)


def quantize_embedding(embedding, maximum, reference_max):
    if not 1 <= maximum <= 255:
        raise ValueError("BMP document impacts must be between 1 and 255.")
    if reference_max <= 0:
        raise ValueError("The quantization reference must be positive.")
    return {
        token: min(maximum, max(1, math.ceil(value * maximum / reference_max)))
        for token, value in embedding.items()
    }


def determine_max_document_impact(query_embeddings, requested_maximum):
    max_query_terms = max(
        (len(merge_embedding(tokens, values)) for _, tokens, values in query_embeddings),
        default=0,
    )
    if max_query_terms == 0:
        return requested_maximum

    safe_maximum = MAX_SCORE // (max_query_terms * MAX_QUERY_WEIGHT)
    if safe_maximum < 1:
        raise ValueError(
            "BMP cannot safely accumulate scores for queries with more than "
            f"{MAX_SCORE // MAX_QUERY_WEIGHT} positive terms."
        )
    return min(requested_maximum, safe_maximum)


def build_index(
    index_path,
    document_embeddings,
    block_size,
    compress_range,
    max_document_impact,
):
    if not 1 <= block_size <= 256:
        raise ValueError("Block size must be between 1 and 256.")
    if not document_embeddings:
        raise ValueError("BMP requires at least one document.")

    merged_documents = [
        (str(doc_id), merge_embedding(tokens, values))
        for doc_id, tokens, values in document_embeddings
    ]
    document_ids = [doc_id for doc_id, _ in merged_documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("BMP requires unique document IDs.")

    reference_max = max(
        (value for _, embedding in merged_documents for value in embedding.values()),
        default=0,
    )
    if reference_max <= 0:
        raise ValueError("BMP requires at least one positive document embedding value.")
    vocabulary = {
        token for _, embedding in merged_documents for token in embedding
    }
    if len(vocabulary) > MAX_VOCABULARY_SIZE:
        raise ValueError(
            "BMP supports at most 65,536 unique document embedding dimensions."
        )

    indexer = Indexer(
        str(index_path),
        bsize=block_size,
        compress_range=compress_range,
    )
    for doc_id, embedding in merged_documents:
        quantized = quantize_embedding(
            embedding,
            max_document_impact,
            reference_max,
        )
        indexer.add_document(doc_id, quantized)
    indexer.finish()

    return BmpIndex(
        searcher=Searcher(str(index_path)),
        vocabulary=frozenset(vocabulary),
        document_ids=frozenset(document_ids),
    )


def retrieve(index, query_embeddings, k, alpha, beta):
    if k < 1:
        raise ValueError("k must be at least 1.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not 0 < beta <= 1:
        raise ValueError("beta must be greater than 0 and at most 1.")

    depth = min(k, len(index.document_ids))
    results = []
    for query_id, tokens, values in query_embeddings:
        query = {
            token: value
            for token, value in merge_embedding(tokens, values).items()
            if token in index.vocabulary
        }
        if not query or depth == 0:
            results.append((str(query_id), []))
            continue

        doc_ids, scores = index.searcher.search(
            query,
            k=depth,
            alpha=alpha,
            beta=beta,
        )
        ranking = sorted(
            (
                (str(doc_id), float(score))
                for doc_id, score in zip(doc_ids, scores)
                if str(doc_id) in index.document_ids and float(score) > 0
            ),
            key=lambda result: result[1],
            reverse=True,
        )[:depth]
        results.append((str(query_id), ranking))
    return results


@retrieve_command()
@click.option(
    "--block-size",
    type=click.IntRange(min=1, max=256),
    default=8,
    show_default=True,
    help="Number of documents per BMP block.",
)
@click.option(
    "--compress-range/--no-compress-range",
    default=True,
    show_default=True,
    help="Compress block range maximum scores.",
)
@click.option(
    "--max-document-impact",
    type=click.IntRange(min=1, max=255),
    default=255,
    show_default=True,
    help="Maximum globally quantized document impact.",
)
@click.option(
    "--alpha",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
    help="BMP block-pruning aggressiveness; 1.0 is rank-safe.",
)
@click.option(
    "--beta",
    type=click.FloatRange(min=0, max=1, min_open=True),
    default=1.0,
    show_default=True,
    help="Fraction of the highest-weight query terms retained.",
)
def main(
    dataset,
    embedding,
    output,
    k,
    block_size,
    compress_range,
    max_document_impact,
    alpha,
    beta,
):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)

    document_embeddings = list(load_embeddings(dataset, embedding, "doc"))
    query_embeddings = list(load_embeddings(dataset, embedding, "query"))
    effective_max_document_impact = determine_max_document_impact(
        query_embeddings,
        max_document_impact,
    )
    register_metadata(
        {
            "actor": {"team": "reneuir-baselines"},
            "tag": (
                f"bmp-{embedding.replace('/', '-')}-{block_size}-"
                f"{compress_range}-{effective_max_document_impact}-"
                f"{alpha}-{beta}-{k}"
            ),
        }
    )

    with TemporaryDirectory() as temporary_directory:
        index_path = Path(temporary_directory) / "index.bmp"
        with tracking(
            export_file_path=output / "index-metadata.yml",
            export_format=ExportFormat.IR_METADATA,
        ):
            index = build_index(
                index_path,
                document_embeddings,
                block_size,
                compress_range,
                effective_max_document_impact,
            )

        with tracking(
            export_file_path=output / "retrieval-metadata.yml",
            export_format=ExportFormat.IR_METADATA,
        ):
            results = retrieve(index, query_embeddings, k, alpha, beta)

    with gzip.open(output / "run.txt.gz", "wt") as run_file:
        for query_id, ranking in results:
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                run_file.write(f"{query_id} Q0 {doc_id} {rank} {score} bmp\n")


if __name__ == "__main__":
    main()
