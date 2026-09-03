#!/usr/bin/env python3
"""Compare datasets by how they rank embeddings by effectiveness (nDCG@10),
using the naive (exact, brute-force) retrieval engine only.

Since effectiveness (nDCG@10) is a property of the embedding, not of the
(approximate) retrieval engine used to search it, the retrieval engine is
fixed to ``naive-search`` here. For every dataset, a table of embeddings is
built, sorted by ``nDCG@10`` (best first). The agreement between the
embedding rankings produced by different datasets is then reported as
Kendall's tau rank correlation and visualized as a heatmap PDF.
"""
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")

import pandas as pd

from plot_retrieval_engine_correlation import (
    DEFAULT_INPUT_FILE,
    DEFAULT_OVERVIEW_FILE,
    load_data,
    plot_correlation_heatmap,
    prettify_embedding_name,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PDF = HERE / "dataset-effectiveness-correlation.pdf"
EFFECTIVENESS_METRIC = "nDCG@10"
NAIVE_ENGINE = "naive-search"

# Datasets excluded from this analysis (e.g. the tiny smoke-test dataset, and
# datasets deemed redundant with others already covered).
EXCLUDED_DATASETS = {
    "tiny-example-20251002_0-training",
    "trec-19-web-20251008-test",
    "trec-20-web-20251008-test",
    "trec-21-web-20251008-test",
    "trec-23-web-20251008-test",
    "trec-29-deep-learning-passages-20250926-training",
    "trec-robust-2004-fold-2-20250926-test",
    "trec-robust-2004-fold-3-20250926-test",
    "trec-robust-2004-fold-4-20250926-test",
    "trec-robust-2004-fold-5-20250926-test",
}

# Human-readable names for the (non-excluded) datasets, used for plot labels.
PRETTY_DATASET_NAMES = {
    "trec-18-web-20251008-test": "Web 2009",
    "trec-22-web-20251008-test": "Web 2013",
    "trec-28-deep-learning-passages-20250926-training": "DL 2019",
    "trec-28-misinfo-20251008_1-test": "Misinfo 2019",
    "trec-33-rag-20250926_1-training": "RAG 2024",
    "trec-robust-2004-fold-1-20250927-test": "Robust04",
}


def prettify_dataset_labels(correlation: pd.DataFrame) -> pd.DataFrame:
    """Rename dataset identifiers to human-readable names and label both axes
    as "Dataset"."""
    pretty = correlation.rename(index=PRETTY_DATASET_NAMES, columns=PRETTY_DATASET_NAMES)
    pretty.index.name = "Dataset"
    pretty.columns.name = "Dataset"
    return pretty


def per_dataset_tables(df: pd.DataFrame, engine: str = NAIVE_ENGINE) -> "dict[str, pd.DataFrame]":
    """Per dataset, sort embeddings by nDCG@10 (best first), using only the
    given (naive) retrieval engine."""
    engine_df = df[df["Retrieval"] == engine]
    columns = ["embedding/model", EFFECTIVENESS_METRIC]
    tables = {}
    for dataset, group in engine_df.groupby("tira-dataset-id"):
        table = (
            group[columns]
            .groupby("embedding/model", as_index=False)
            .mean()
            .sort_values(EFFECTIVENESS_METRIC, ascending=False)
            .reset_index(drop=True)
        )
        tables[dataset] = table
    return tables


def effectiveness_matrix(df: pd.DataFrame, engine: str = NAIVE_ENGINE) -> pd.DataFrame:
    """One row per embedding/model, one column per dataset."""
    engine_df = df[df["Retrieval"] == engine]
    return engine_df.pivot_table(
        index="embedding/model",
        columns="tira-dataset-id",
        values=EFFECTIVENESS_METRIC,
        aggfunc="mean",
    )


def _escape_latex(value: str) -> str:
    return str(value).replace("_", r"\_")


def datasets_top_k_latex_table(tables: "dict[str, pd.DataFrame]", k: int = 3) -> str:
    """Build a LaTeX table with columns Dataset, Top-1 Model, Top-1 nDCG@10,
    ..., Top-k Model, Top-k nDCG@10, one row per dataset."""
    col_spec = "l" + "lr" * k
    rank_header = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{Top-{rank}}}" for rank in range(1, k + 1)
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(k)
    )
    subheader = " & ".join(["Model", EFFECTIVENESS_METRIC] * k)

    rows = []
    for dataset in sorted(tables):
        table = tables[dataset]
        row = [_escape_latex(dataset)]
        for rank in range(k):
            if rank < len(table):
                model = _escape_latex(
                    prettify_embedding_name(table.loc[rank, "embedding/model"])
                )
                score = table.loc[rank, EFFECTIVENESS_METRIC]
                row += [model, f"{score:.3f}"]
            else:
                row += ["--", "--"]
        rows.append(" & ".join(row) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Dataset & {rank_header} \\\\",
        cmidrules,
        f" & {subheader} \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Top-{k} most effective embeddings ({EFFECTIVENESS_METRIC}) per"
        f" dataset, using the {_escape_latex(NAIVE_ENGINE)} retrieval engine.}}",
        r"\label{tab:top3-models-by-dataset}",
        r"\end{table}",
    ]
    return "\n".join(lines)


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_INPUT_FILE,
    show_default=True,
    help="The evaluation .jsonl.gz file to analyze.",
)
@click.option(
    "--overview-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_OVERVIEW_FILE,
    show_default=True,
    help="The dataset overview.json used to derive per-query retrieval time"
    " if retrieval_per_query.runtime_wallclock is not already present.",
)
@click.option(
    "--output-pdf",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT_PDF,
    show_default=True,
    help="Where to write the Kendall's tau correlation heatmap PDF.",
)
@click.option(
    "--tables-output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional directory to also write the per-dataset sorted ranking tables to as CSV.",
)
@click.option(
    "--top3-tex-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=HERE / "top3-models-by-dataset.tex",
    show_default=True,
    help="Where to write the LaTeX table with the top-3 models per dataset.",
)
def main(
    input_file: Path,
    overview_file: Path,
    output_pdf: Path,
    tables_output_dir: "Path | None",
    top3_tex_file: Path,
):
    df = load_data(input_file, overview_file)
    df = df[~df["tira-dataset-id"].isin(EXCLUDED_DATASETS)].reset_index(drop=True)

    tables = per_dataset_tables(df)
    for dataset, table in tables.items():
        click.echo(
            f"\n=== Dataset: {dataset} (embeddings sorted by {EFFECTIVENESS_METRIC},"
            f" retrieval engine: {NAIVE_ENGINE}) ==="
        )
        click.echo(table.to_string(index=False))
        if tables_output_dir is not None:
            tables_output_dir.mkdir(parents=True, exist_ok=True)
            csv_file = tables_output_dir / f"{dataset}-effectiveness-ranking.csv"
            table.to_csv(csv_file, index=False)
            click.echo(f"Wrote {csv_file}")

    correlation = effectiveness_matrix(df).corr(method="kendall")
    correlation.index.name = "Dataset"
    correlation.columns.name = "Dataset"

    click.echo("\n=== Kendall's tau correlation between datasets ===")
    click.echo(correlation.round(3).to_string())

    plot_correlation_heatmap(
        prettify_dataset_labels(correlation),
        output_pdf,
        title="Correlation of effective embeddings per dataset",
        prettify=False,
        lower_triangle_only=True,
    )
    click.echo(f"\nWrote {output_pdf}")

    latex_table = datasets_top_k_latex_table(tables)
    click.echo("\n=== Top-3 models per dataset (LaTeX) ===")
    click.echo(latex_table)
    top3_tex_file.write_text(latex_table + "\n")
    click.echo(f"\nWrote {top3_tex_file}")


if __name__ == "__main__":
    main()
