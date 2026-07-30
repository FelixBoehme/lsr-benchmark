from unittest.mock import MagicMock, patch

import pytest

from lsr_benchmark._commands import _evaluate as evaluator

MODULE = "lsr_benchmark._commands._evaluate"


@patch(f"{MODULE}.Path.is_dir", return_value=True)
@patch(f"{MODULE}.lines_if_valid")
@patch(f"{MODULE}.Path.is_file", return_value=True)
@patch(f"{MODULE}.Path.read_text", return_value="dummy_run_content")  # Mock read_text here!
@patch(f"{MODULE}.ir_measures.read_trec_run", return_value=["run_data"])
def test_new_metadata_parsing(mock_read_run, mock_read_text, mock_is_file, mock_lines, mock_is_dir):
    mock_lines.return_value = [
        {"name": "myapproach-doc-123", "content": "doc_meta"},
        {"name": "myapproach-query-123", "content": "query_meta"},
        {"name": "myapproach-other-123", "content": "standard_meta"},
    ]

    metadata, _ = evaluator.__read_metrics("dummy_dir")

    assert metadata["myapproach-doc"] == "doc_meta"
    assert metadata["myapproach-query"] == "query_meta"
    assert metadata["myapproach"] == "standard_meta"


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, ["run"]))
@patch(f"{MODULE}.__get_dataset_name", return_value="dataset-1")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@patch(f"{MODULE}.ir_measures.calc_aggregate", return_value={"P@10": 0.85})
def test_original_aggregated_evaluation(mock_calc_agg, mock_lsr, mock_get_emb, mock_get_ds, mock_read):
    mock_lsr.load.return_value.has_qrels.return_value = True

    measure_mock = MagicMock()
    measure_mock.__str__.return_value = "P@10"

    result = evaluator.evaluate_approach("dummy", [("P_10", "ir_measure", measure_mock)], per_query=False)

    assert result["P@10"] == 0.85
    assert result["tira-dataset-id"] == "dataset-1"
    assert "micro-averages" not in result
    assert "macro-averages" not in result


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, ["run"]))
@patch(f"{MODULE}.__get_dataset_name", return_value="joint_dataset")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@patch(f"{MODULE}.ir_measures.calc")
@patch.dict(f"{MODULE}.JOINT_TO_DATASETS", {"joint_dataset": {"datasets": ["sub1", "sub2"]}}, clear=True)
def test_new_per_query_and_joint_evaluation(mock_calc, mock_lsr, mock_get_emb, mock_get_ds, mock_read):
    mock_lsr.load.return_value.has_qrels.return_value = True

    measure_mock = MagicMock()
    measure_mock.__str__.return_value = "P@10"

    class MockMetric:
        def __init__(self, val, qid):
            self.measure = measure_mock
            self.value = val
            self.query_id = qid

    mock_calc.side_effect = [
        MagicMock(aggregated={measure_mock: 0.2}, per_query=[MockMetric(0.2, "q1")]),
        MagicMock(
            aggregated={measure_mock: 0.8},
            per_query=[MockMetric(0.8, "q2"), MockMetric(0.8, "q3"), MockMetric(0.8, "q4")],
        ),
    ]

    result = evaluator.evaluate_approach("dummy", [("P_10", "ir_measure", measure_mock)], per_query=True)

    assert result["sub1"]["P@10"]["q1"] == 0.2
    assert result["sub2"]["P@10"]["q3"] == 0.8

    # Micro Average: (0.2 + 0.8 + 0.8 + 0.8) / 4 queries = 0.65
    assert result["micro-averages"]["P@10"] == pytest.approx(0.65)

    # Macro Average: (0.2 + 0.8) / 2 datasets = 0.5
    assert result["macro-averages"]["P@10"] == pytest.approx(0.5)
