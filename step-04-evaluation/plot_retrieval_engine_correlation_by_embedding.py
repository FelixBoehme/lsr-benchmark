#!/usr/bin/env python3
"""Compare retrieval engines by how they rank embeddings by per-query
retrieval time, averaged across all datasets.

This is a variant of ``plot_retrieval_engine_correlation.py``: instead of
treating every dataset + embedding combination as a separate sample, the
``retrieval_per_query.runtime_wallclock`` is first averaged per retrieval
engine and embedding/model across all datasets. For every retrieval engine, a
table of embeddings is built, sorted by this average runtime (fastest first).
The agreement between the rankings produced by different retrieval engines is
then reported as Kendall's tau rank correlation and visualized as a heatmap
PDF.
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
DEFAULT_OUTPUT_PDF = HERE / "retrieval-engine-runtime-correlation-by-embedding.pdf"
DEFAULT_TOP3_TEX_FILE = HERE / "top3-embeddings-by-engine.tex"
TOP3_ENGINES = ["kannolo", "pyterrier-splade-pisa", "seismic"]


def average_runtime_per_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """Average retrieval_per_query.runtime_wallclock per retrieval engine and
    embedding/model across all datasets."""
    return df.groupby(["Retrieval", "embedding/model"], as_index=False)[
        "retrieval_per_query.runtime_wallclock"
    ].mean()


def per_engine_tables(averaged: pd.DataFrame) -> "dict[str, pd.DataFrame]":
    """Per retrieval engine, sort embeddings by their average
    retrieval_per_query.runtime_wallclock across all datasets (fastest first)."""
    tables = {}
    for engine, group in averaged.groupby("Retrieval"):
        table = (
            group[["embedding/model", "retrieval_per_query.runtime_wallclock"]]
            .sort_values("retrieval_per_query.runtime_wallclock")
            .reset_index(drop=True)
        )
        tables[engine] = table
    return tables


def runtime_matrix(averaged: pd.DataFrame) -> pd.DataFrame:
    """One row per embedding/model, one column per retrieval engine."""
    return averaged.pivot_table(
        index="embedding/model",
        columns="Retrieval",
        values="retrieval_per_query.runtime_wallclock",
        aggfunc="mean",
    )


def _escape_latex(value: str) -> str:
    return str(value).replace("_", r"\_")


def top3_embeddings_latex_table(
    tables: "dict[str, pd.DataFrame]",
    engines: "list[str]" = TOP3_ENGINES,
    k: int = 3,
) -> str:
    """Build a LaTeX table with the top-k fastest embeddings (and their
    average retrieval_per_query.runtime_wallclock) for the given engines."""
    missing = [engine for engine in engines if engine not in tables]
    if missing:
        raise ValueError(f"No data found for retrieval engine(s): {missing}")

    col_spec = "l" + "lr" * len(engines)
    header_engines = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{_escape_latex(engine)}}}" for engine in engines
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(engines))
    )
    subheader = " & ".join(["Embedding", "ms/query"] * len(engines))

    rows = []
    for rank in range(k):
        row = [str(rank + 1)]
        for engine in engines:
            table = tables[engine]
            if rank < len(table):
                embedding = _escape_latex(
                    prettify_embedding_name(table.loc[rank, "embedding/model"])
                )
                runtime = table.loc[rank, "retrieval_per_query.runtime_wallclock"]
                row += [embedding, f"{runtime:.3f}"]
            else:
                row += ["--", "--"]
        rows.append(" & ".join(row) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Rank & {header_engines} \\\\",
        cmidrules,
        f" & {subheader} \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Top-{k} fastest embeddings (average retrieval time per query,"
        " ms) for "
        + ", ".join(_escape_latex(engine) for engine in engines)
        + ".}",
        r"\label{tab:top3-embeddings-by-engine}",
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
    help="Optional directory to also write the per-engine sorted ranking tables to as CSV.",
)
@click.option(
    "--top3-tex-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_TOP3_TEX_FILE,
    show_default=True,
    help="Where to write the LaTeX table with the top-3 fastest embeddings"
    f" for {', '.join(TOP3_ENGINES)}.",
)
def main(
    input_file: Path,
    overview_file: Path,
    output_pdf: Path,
    tables_output_dir: "Path | None",
    top3_tex_file: Path,
):
    df = load_data(input_file, overview_file)
    averaged = average_runtime_per_embedding(df)

    tables = per_engine_tables(averaged)
    for engine, table in tables.items():
        click.echo(
            f"\n=== Retrieval engine: {engine} "
            "(embeddings sorted by average retrieval_per_query.runtime_wallclock"
            " across all datasets) ==="
        )
        click.echo(table.to_string(index=False))
        if tables_output_dir is not None:
            tables_output_dir.mkdir(parents=True, exist_ok=True)
            csv_file = tables_output_dir / f"{engine}-avg-runtime-ranking.csv"
            table.to_csv(csv_file, index=False)
            click.echo(f"Wrote {csv_file}")

    correlation = runtime_matrix(averaged).corr(method="kendall")

    click.echo(
        "\n=== Kendall's tau correlation between retrieval engines"
        " (based on per-embedding/model average runtime) ==="
    )
    click.echo(correlation.round(3).to_string())

    plot_correlation_heatmap(correlation, output_pdf, lower_triangle_only=True)
    click.echo(f"\nWrote {output_pdf}")

    latex_table = top3_embeddings_latex_table(tables, TOP3_ENGINES)
    click.echo(
        f"\n=== Top-3 fastest embeddings for {', '.join(TOP3_ENGINES)} (LaTeX) ==="
    )
    click.echo(latex_table)
    top3_tex_file.write_text(latex_table + "\n")
    click.echo(f"\nWrote {top3_tex_file}")


if __name__ == "__main__":
    main()
