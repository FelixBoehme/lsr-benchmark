import sqlite3


def create_connection():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("""
        CREATE TABLE index_terms (
            term_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            score REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TEMP TABLE query_terms (
            term_id TEXT NOT NULL,
            score REAL NOT NULL
        )
    """)
    return conn


def _score_value(score, quantize):
    score = float(score)
    if quantize:
        return int(round(score * 100))
    return score


def index_documents(conn, doc_embeddings, quantize=False):
    with conn:
        for doc_id, tokens, values in doc_embeddings:
            rows = [
                (str(term_id), doc_id, _score_value(score, quantize))
                for term_id, score in zip(tokens, values)
            ]
            conn.executemany(
                "INSERT INTO index_terms (term_id, doc_id, score) VALUES (?, ?, ?)",
                rows,
            )

    conn.execute("CREATE INDEX index_terms_term_doc_idx ON index_terms (term_id, doc_id)")


def retrieve_query(conn, query_id, tokens, values, k):
    conn.execute("DELETE FROM query_terms")
    conn.executemany(
        "INSERT INTO query_terms (term_id, score) VALUES (?, ?)",
        [(str(term_id), float(score)) for term_id, score in zip(tokens, values)],
    )
    results = conn.execute(
        """
        SELECT
            ? AS query_id,
            SUM(index_terms.score * query_terms.score) AS score,
            index_terms.doc_id
        FROM index_terms
        JOIN query_terms USING (term_id)
        GROUP BY index_terms.doc_id
        HAVING SUM(index_terms.score * query_terms.score) > 0
        ORDER BY score DESC, index_terms.doc_id ASC
        LIMIT ?
        """,
        (query_id, k),
    )
    return [(qid, float(score), doc_id) for qid, score, doc_id in results.fetchall()]
