from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ir_datasets.formats import TrecQrel
from ir_measures import ScoredDoc, parse_measure

from lsr_benchmark._commands import _evaluate as evaluator

MODULE = "lsr_benchmark._commands._evaluate"


@patch(f"{MODULE}.Path.is_dir", return_value=True)
@patch(f"{MODULE}.lines_if_valid")
@patch(f"{MODULE}.Path.is_file", return_value=True)
@patch(f"{MODULE}.Path.read_text", return_value="dummy_run_content")
@patch(f"{MODULE}.ir_measures.read_trec_run", return_value=["run_data"])
def test_metadata_parsing(mock_read_run, mock_read_text, mock_is_file, mock_lines, mock_is_dir):
    mock_lines.return_value = [
        {"name": "myapproach-doc-123", "content": "doc_meta"},
        {"name": "myapproach-query-123", "content": "query_meta"},
        {"name": "myapproach-other-123", "content": "standard_meta"},
    ]

    metadata, _ = evaluator.__read_metrics("dummy_dir")

    assert metadata["myapproach-doc"] == "doc_meta"
    assert metadata["myapproach-query"] == "query_meta"
    assert metadata["myapproach"] == "standard_meta"


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, [ScoredDoc("q1", "d1", 1.0)]))
@patch(f"{MODULE}.__get_dataset_name", return_value="dataset-1")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_aggregated_evaluation(mock_lsr, mock_get_emb, mock_get_ds, mock_read):
    mock_lsr.load.return_value.has_qrels.return_value = True
    mock_lsr.load.return_value.qrels = [TrecQrel(query_id="q1", doc_id="d1", relevance=1, iteration=0)]

    measure = parse_measure("P@10")

    result = evaluator.evaluate_approach("dummy", [("P_10", "ir_measure", measure)], per_query=False)

    assert result["P@10"] == pytest.approx(0.1)
    assert result["tira-dataset-id"] == "dataset-1"
    assert "micro-averages" not in result
    assert "macro-averages" not in result


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, ["run"]))
@patch(f"{MODULE}.__get_dataset_name", return_value="joint_dataset")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@patch(f"{MODULE}.ir_measures.calc")
@patch.dict(
    f"{MODULE}.JOINT_TO_DATASETS",
    {
        "joint_dataset": {
            "datasets": ["sub1", "sub2"],
            "settings": SimpleNamespace(query=None),
        }
    },
    clear=True,
)
def test_per_query_and_joint_evaluation(mock_calc, mock_lsr, mock_get_emb, mock_get_ds, mock_read):
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
            per_query=[
                MockMetric(0.8, "q2"),
                MockMetric(0.8, "q3"),
                MockMetric(0.8, "q4"),
            ],
        ),
    ]

    result = evaluator.evaluate_approach("dummy", [("P_10", "ir_measure", measure_mock)], per_query=True)

    assert result["sub1"]["P@10"]["q1"] == 0.2
    assert result["sub2"]["P@10"]["q3"] == 0.8

    # Micro Average: (0.2 + 0.8 + 0.8 + 0.8) / 4 queries = 0.65
    assert result["micro-averages"]["P@10"] == pytest.approx(0.65)

    # Macro Average: (0.2 + 0.8) / 2 datasets = 0.5
    assert result["macro-averages"]["P@10"] == pytest.approx(0.5)


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, ["run"]))
@patch(f"{MODULE}.__get_dataset_name", return_value="joint_dataset")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@patch(f"{MODULE}.ir_measures.calc")
@patch.dict(
    f"{MODULE}.JOINT_TO_DATASETS",
    {
        "joint_dataset": {
            "datasets": ["sub1", "sub2"],
            "settings": SimpleNamespace(
                query=evaluator.DuplicateBehaviour.PREFIX,
            ),
        }
    },
    clear=True,
)
def test_per_query_joint_prefixes_constituent_query_ids(mock_calc, mock_lsr, mock_get_emb, mock_get_ds, mock_read):
    joint_dataset = MagicMock()
    joint_dataset.has_qrels.return_value = True

    sub1 = MagicMock()
    sub1.qrels = [TrecQrel(query_id="q1", doc_id="doc1", relevance=1, iteration=0)]

    sub2 = MagicMock()
    sub2.qrels = [TrecQrel(query_id="q2", doc_id="doc2", relevance=1, iteration=0)]

    mock_lsr.load.side_effect = [joint_dataset, sub1, sub2]

    measure_mock = MagicMock()
    measure_mock.__str__.return_value = "P@10"

    class MockMetric:
        def __init__(self, val, qid):
            self.measure = measure_mock
            self.value = val
            self.query_id = qid

    observed_query_ids = []

    def calc_side_effect(_measures, qrels, _run):
        query_ids = [qrel.query_id for qrel in qrels]
        observed_query_ids.append(query_ids)
        value = 0.2 if query_ids[0].startswith("d0-") else 0.8
        return MagicMock(
            aggregated={measure_mock: value},
            per_query=[MockMetric(value, query_id) for query_id in query_ids],
        )

    mock_calc.side_effect = calc_side_effect

    result = evaluator.evaluate_approach("dummy", [("P_10", "ir_measure", measure_mock)], per_query=True)

    assert observed_query_ids == [["d0-q1"], ["d1-q2"]]
    assert result["sub1"]["P@10"]["d0-q1"] == 0.2
    assert result["sub2"]["P@10"]["d1-q2"] == 0.8


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, ["run"]))
@patch(f"{MODULE}.__get_dataset_name", return_value="standalone_dataset")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@patch(f"{MODULE}.ir_measures.calc")
@patch.dict(f"{MODULE}.JOINT_TO_DATASETS", {}, clear=True)
def test_per_query_standalone_keeps_original_query_ids(mock_calc, mock_lsr, mock_get_emb, mock_get_ds, mock_read):
    standalone = MagicMock()
    standalone.has_qrels.return_value = True
    standalone.qrels = [TrecQrel(query_id="q1", doc_id="doc1", relevance=1, iteration=0)]
    mock_lsr.load.return_value = standalone

    measure_mock = MagicMock()
    measure_mock.__str__.return_value = "P@10"

    metric = MagicMock()
    metric.measure = measure_mock
    metric.value = 0.4
    metric.query_id = "q1"
    mock_calc.return_value = MagicMock(
        aggregated={measure_mock: 0.4},
        per_query=[metric],
    )

    result = evaluator.evaluate_approach("dummy", [("P_10", "ir_measure", measure_mock)], per_query=True)

    passed_qrels = mock_calc.call_args.args[1]
    assert [qrel.query_id for qrel in passed_qrels] == ["q1"]
    assert result["standalone_dataset"]["P@10"]["q1"] == 0.4
    assert "micro-averages" not in result
    assert "macro-averages" not in result
