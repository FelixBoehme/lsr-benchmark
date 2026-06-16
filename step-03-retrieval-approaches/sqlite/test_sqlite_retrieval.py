import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "sqlite_retrieval.py"
MODULE_SPEC = importlib.util.spec_from_file_location("sqlite_retrieval", MODULE_PATH)
sqlite_retrieval = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(sqlite_retrieval)

create_connection = sqlite_retrieval.create_connection
index_documents = sqlite_retrieval.index_documents
retrieve_query = sqlite_retrieval.retrieve_query


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.conn = create_connection()

    def tearDown(self):
        self.conn.close()

    def test_01_identical_vector_is_top_result(self):
        index_documents(
            self.conn,
            [("d1", ["1"], [1.0]), ("d2", ["1"], [0.5]), ("d3", ["2"], [1.0])],
        )
        results = retrieve_query(self.conn, "q1", ["1"], [1.0], k=3)
        self.assertEqual(results[0][2], "d1")

    def test_02_topk_is_respected(self):
        index_documents(
            self.conn,
            [
                ("d1", ["1"], [1.0]),
                ("d2", ["1"], [0.8]),
                ("d3", ["1"], [0.6]),
                ("d4", ["1"], [0.4]),
            ],
        )
        results = retrieve_query(self.conn, "q1", ["1"], [1.0], k=2)
        self.assertLessEqual(len(results), 2)

    def test_03_scores_sorted_descending(self):
        index_documents(
            self.conn,
            [("d1", ["1"], [0.5]), ("d2", ["1"], [1.0]), ("d3", ["1"], [0.0])],
        )
        results = retrieve_query(self.conn, "q1", ["1"], [1.0], k=3)
        scores = [result[1] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_04_query_id_correct(self):
        index_documents(self.conn, [("d1", ["1"], [1.0])])
        results = retrieve_query(self.conn, "my-query", ["1"], [1.0], k=1)
        self.assertEqual(results[0][0], "my-query")

    def test_05_doc_id_correct(self):
        index_documents(self.conn, [("best-doc", ["1"], [1.0]), ("worst-doc", ["2"], [1.0])])
        results = retrieve_query(self.conn, "q1", ["1"], [1.0], k=2)
        self.assertEqual(results[0][2], "best-doc")

    def test_06_multiple_queries(self):
        index_documents(self.conn, [("d1", ["1"], [1.0]), ("d2", ["2"], [1.0])])
        results_q1 = retrieve_query(self.conn, "q1", ["1"], [1.0], k=1)
        results_q2 = retrieve_query(self.conn, "q2", ["2"], [1.0], k=1)
        self.assertEqual(results_q1[0][2], "d1")
        self.assertEqual(results_q2[0][2], "d2")

    def test_07_score_of_orthogonal_vectors_is_0(self):
        index_documents(self.conn, [("d1", ["2"], [1.0])])
        results = retrieve_query(self.conn, "q1", ["1"], [1.0], k=1)
        self.assertEqual(results, [])

    def test_08_quantized_index_still_retrieves_best_document(self):
        index_documents(
            self.conn,
            [("d1", ["1"], [0.91]), ("d2", ["1"], [0.12])],
            quantize=True,
        )
        results = retrieve_query(self.conn, "q1", ["1"], [1.0], k=2)
        self.assertEqual(results[0][2], "d1")
        self.assertEqual(results[0][1], 91.0)


if __name__ == "__main__":
    unittest.main()
