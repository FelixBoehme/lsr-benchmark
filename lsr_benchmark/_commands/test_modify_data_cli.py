import click
import pytest

from lsr_benchmark.datasets import all_datasets
from ._modify_data import DATSET_TO_MAPPING

def test_mapping_for_all_datasets():
    unmapped = [d for d in all_datasets() if d not in DATSET_TO_MAPPING]
    assert not unmapped, f"Mappings missing for: {unmapped}"
