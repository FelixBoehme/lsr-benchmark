import click
import pytest

from lsr_benchmark.datasets import all_datasets

from ._modify_data import DATASET_TO_MAPPING, modify_data


def test_mapping_for_all_datasets():
    unmapped = [d for d in all_datasets() if d not in DATASET_TO_MAPPING]
    assert not unmapped, f"Mappings missing for: {unmapped}"


def test_fail_unknown_dataset():
    with pytest.raises(click.BadParameter, match="unknown-dataset"):
        modify_data(["unknown-dataset"], None, True, None, None)
