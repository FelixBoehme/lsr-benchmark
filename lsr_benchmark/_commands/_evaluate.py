import gzip
import logging
import re
from copy import deepcopy
from glob import glob
from gzip import GzipFile
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping
from zipfile import ZipFile

import click
import ir_measures
import pandas as pd
import yaml
from ir_measures import ScoredDoc, parse_trec_measure
from ir_datasets.formats import TrecQrel
from tira.check_format import lines_if_valid

import lsr_benchmark
from lsr_benchmark.datasets import TIRA_DATASET_ID_TO_IR_DATASET_ID, all_embeddings

from ._modify_data import DATASET_TO_MAPPING, MAPPING_TO_DATASET

if TYPE_CHECKING:
    from typing import _KT, _T, _VT, Any, Callable, Literal, Optional, Union

    from ir_measures import Measure, ScoredDoc

    Metadata = dict[str, Any]


def __get_nested(
    d: "Mapping[_KT, Union[dict, _VT]]", keys: "list[_KT]"
) -> "Union[Mapping[_KT, Union[dict, _VT]], _VT]":
    """Recursively retrieves a value from a nested mapping using a list of keys.

    Args:
        d (Mapping[_KT, Union[dict, _VT]]): The dictionary to traverse.
        keys (list[_KT]): A list of keys representing the path to the desired value.

    Raises:
        TypeError: If an intermediate value in the path is not a mapping.
        KeyError: If any key in the path is not found in the corresponding mapping.

    Returns:
        The value located at the nested key path.
    """
    out: "Union[Mapping[_KT, Union[dict, _VT]], _VT]" = d
    for i, key in enumerate(keys):
        if not isinstance(out, Mapping):
            raise TypeError(f"The value at {'>'.join(map(str, keys[:i]))} is not a mapping. Have {out}.")
        if key not in out:
            raise KeyError(f"The key {'>'.join(map(str, keys[:i+1]))} could not be found. Have {out}.")
        out = out[key]
    return out


def __get_nested_or_default(
    d: "Mapping[_KT, Union[dict, _VT]]", keys: "list[_KT]", default: "_T" = None
) -> "Union[_VT, _T]":
    try:
        return __get_nested(d, keys)
    except KeyError:
        return default


def __read_metrics(name: str) -> "tuple[dict[str, Metadata], list[ScoredDoc]]":
    metadata: "dict[str, Metadata]" = {}

    if name.endswith('/run.txt.gz'):
        name = name.replace('/run.txt.gz', '/')

    if Path(name).is_dir():
        for line in lines_if_valid(Path(name), "ir_metadata"):
            metadata[line['name'].replace('.', '').split("-")[0]] = line['content']
        if (Path(name) / "run.txt").is_file():
            run = list(ir_measures.read_trec_run((Path(name) / "run.txt").read_text()))
        else:
            run = list(ir_measures.read_trec_run(gzip.open(Path(name) / "run.txt.gz", "rt")))
    else:
        with ZipFile(name) as archive:
            for entry in archive.filelist:
                if (m := re.match(r"(\w+)-metadata.ya?ml", entry.filename)) is not None:
                    with archive.open(entry) as file:
                        metadata[m.group(1)] = yaml.safe_load(file)
            with archive.open("run.txt.gz", mode="r") as compressed:
                with GzipFile(fileobj=compressed, mode="r") as binary:
                    with TextIOWrapper(binary, encoding="utf-8") as file:
                        run = list(ir_measures.read_trec_run(file))

    if len(metadata) == 0:
        raise ValueError("I could not read any metadata")
    if len(run) == 0:
        raise ValueError("I could not load a run")
    return metadata, run


def __get_runtime(metadata: "Metadata", param: "Literal['system', 'user', 'wallclock']" = "wallclock") -> "Optional[str]":
    return __get_nested_or_default(metadata, ("resources", "runtime", param))


def __get_energy_usage(metadata: "Metadata", param: "Literal['total', 'cpu', 'gpu', 'ram']" = "total") -> "Optional[float]":
    def __get_energy(device: str) -> float:
        try:
            energy_str = metadata["resources"][device]["energy used system"]
        except KeyError:
            logging.warning(f"Energy for {device} was not reported; using 0 Joules")
        try:
            if match := re.match(r'^(\S+)\s*J$', energy_str):
                return float(match[1])
            else:
                raise ValueError
        except ValueError:
            logging.error(f"Could not parse energy string: '{energy_str}'")
        return 0
    if param == "total":
        return __get_energy("cpu") + __get_energy("gpu") + __get_energy("ram")
    return __get_energy(param)


def __get_avg_cpu_usage(metadata: "Metadata") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "cpu", "used process", "avg"))


def __get_max_ram_usage(metadata: "Metadata") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "ram", "used process", "max"))


def __get_max_vram_usage(metadata: "Metadata") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "gpu", "used vram process", "max"))


def __get_avg_gpu_usage(metadata: "Metadata") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "gpu", "used process", "avg"))


def __get_cpu_temperature(metadata: "Metadata", param: "Literal['avg', 'min', 'max']" = "avg") -> "Optional[int]":
    return __get_nested_or_default(metadata, ("resources", "cpu", "temperature", param))


__efficiency_measures: "dict[str, Callable]" = {
    "runtime": __get_runtime,
    "energy": __get_energy_usage,
    "cpu": __get_avg_cpu_usage,
    "ram": __get_max_ram_usage,
    "gpu": __get_avg_gpu_usage,
    "vram": __get_max_vram_usage,
    "temperature": __get_cpu_temperature,
}


def __parse_tirex_measure(measure: "str") -> "Callable":
    name, *arg = measure.split('_', 2)
    func = __efficiency_measures[name]
    return lambda x: func(x, *arg)


def __parse_measure(measure: "str") -> "tuple[str, Literal['ir_measure', 'tirex'], Measure | Callable]":
    try:
        return (measure, 'ir_measure', parse_trec_measure(measure)[0])
    except (ValueError, NameError):
        # Fall back to non-TREC measures.
        try:
            return (measure, 'ir_measure', ir_measures.parse_measure(measure))
        except (ValueError, NameError):
            # Fall back to TIREx measures.
            return (measure, 'tirex', __parse_tirex_measure(measure))


def __get_dataset_name(metadata: Dict[str, Any]) -> str:
    candidates = set()

    for k, m in metadata.items():
        if "data" in m and "test collection" in m["data"] and "name" in m["data"]["test collection"] and m["data"]["test collection"]["name"]:
            candidates.add(m["data"]["test collection"]["name"])

    candidates = [i for i in candidates if i != '/tira-data/input']
    if len(candidates) != 1:
        raise ValueError(f"I can not extract the dataset from the metadata. I found candidates: {list(candidates)}")

    return list(candidates)[0]


def __get_embedding_name(p: Path):
    # FIXME read this from metadata
    ret = []
    for embedding in all_embeddings():
        if embedding in str(Path(p)).split("/"):
            ret += [embedding]
    if '/none/' in str(p):
        return None
    if len(ret) != 1:
        return None
    return ret[0]


def __get_output_routine(specifier: str) -> "Callable[[pd.DataFrame], None]":
    suffix_to_routine: "dict[str, Callable[[pd.DataFrame], None]]" = {
        ".csv": lambda df: df.to_csv(specifier),
        ".xlsx": lambda df: df.to_excel(specifier),
        ".htm": lambda df: df.to_html(specifier),
        ".html": lambda df: df.to_html(specifier),
        ".json": lambda df: df.to_json(specifier),
        ".gz": lambda df: df.to_json(specifier, lines=True, orient="records"),
        ".tex": lambda df: df.to_latex(specifier),
        ".md": lambda df: df.to_markdown(specifier),
        ".parquet": lambda df: df.to_parquet(specifier),
    }

    if specifier == "-":
        return lambda i: print(pd.DataFrame({j["approach"]: j.to_dict() for _, j in i.iterrows()}))
    elif (routine := suffix_to_routine.get(Path(specifier).suffix, None)) is not None:
        return routine
    else:
        raise ValueError(f"The suffix of {specifier} is not known.")


def __split_run_by_dataset(
    run: "list[ScoredDoc]", mappings: "list[str]"
) -> "dict[str, list[ScoredDoc]]":
    """Splits a joint run into per-dataset runs by stripping the dataset-prefix from doc/query IDs.

    For a joint run, each doc_id and query_id is expected to be prefixed with the dataset mapping
    key followed by a '-' (e.g. "d1-docABC"). The prefix is used to route each scored document
    into the corresponding per-dataset sub-run, and the prefix is stripped from the IDs.

    Args:
        run: The full list of ScoredDoc from the joint run.
        mappings: The list of dataset mapping keys (e.g. ["d1", "d2"]).

    Returns:
        A dict mapping each dataset mapping key to its list of (prefix-stripped) ScoredDocs.
    """
    split: "dict[str, list[ScoredDoc]]" = {m: [] for m in mappings}
    for r in run:
        mapping, _, doc_id = r.doc_id.partition("-")
        _, _, query_id = r.query_id.partition("-")
        if mapping in split:
            split[mapping].append(ScoredDoc(doc_id=doc_id, query_id=query_id, score=r.score))
    return split


def __build_mapped_qrels(datasets: "list[str]") -> "list[TrecQrel]":
    """Builds a unified qrel list with prefixed IDs for micro-average evaluation.

    Each qrel's query_id and doc_id is prefixed with the dataset's mapping key so that they
    align with the original (un-stripped) joint run IDs.

    Args:
        datasets: List of dataset names (tira-dataset-ids).

    Returns:
        A flat list of TrecQrel objects with prefixed IDs from all datasets.
    """
    mapped_qrels = []
    for dataset in datasets:
        mapping = DATASET_TO_MAPPING[dataset]
        dset = lsr_benchmark.load(dataset)
        for q in dset.qrels:
            mapped_qrels.append(TrecQrel(
                query_id=f"{mapping}-{q.query_id}",
                doc_id=f"{mapping}-{q.doc_id}",
                relevance=q.relevance,
                iteration=q.iteration,
            ))
    return mapped_qrels


def __compute_macro_average(
    per_dataset_results: "list[dict[str, Any]]",
    ir_measure_names: "set[str]",
) -> "dict[str, Any]":
    """Computes the macro average of IR measures across all per-dataset result dicts.

    Non-numeric and non-IR-measure fields are ignored.

    Args:
        per_dataset_results: List of result dicts, one per dataset.
        ir_measure_names: Set of IR measure name strings to average.

    Returns:
        A dict of macro-averaged values keyed by "{measure_name}(macro_avg)".
    """
    macro: "dict[str, Any]" = {}
    for name in ir_measure_names:
        values = [r[name] for r in per_dataset_results if name in r and r[name] is not None]
        if values:
            macro[f"{name}(macro_avg)"] = sum(values) / len(values)
    return macro


def evaluate_approach(approach: str, measure: "list[tuple]") -> "list[dict[str, Any]]":
    """Evaluates a single retrieval approach and returns per-dataset (and averaged) results.

    For simple (non-joint) runs, returns a single result dict. For joint (concatenated-dataset)
    runs, returns one result dict per constituent dataset plus two additional rows:
      - A micro-average row: IR measures computed over the full joint run against merged qrels.
      - A macro-average row: arithmetic mean of each IR measure across the per-dataset rows.

    Args:
        approach: Path to the run directory or zip file.
        measure: List of parsed measure tuples as produced by __parse_measure.

    Returns:
        A list of result dicts, each suitable for a row in the output DataFrame.
    """
    efficiency_results: "dict[str, Any]" = {}
    metadata, run = __read_metrics(approach)

    for group, meta in metadata.items():
        for name, typ, func in measure:
            if typ == 'tirex':
                val = func(meta)
                if val is None:
                    logging.warning(
                        f"Measure {name} could not be reported for {approach}.{group} as its metadata is not present"
                    )
                efficiency_results[f"{group}.{name}"] = val

    ir_measures_set = {m for _, t, m in measure if t == 'ir_measure'}
    ir_measure_names = {str(m) for m in ir_measures_set}

    dataset_id = __get_dataset_name(metadata)
    mapping_keys = dataset_id.split("-")
    is_joint = len(mapping_keys) > 1 and all(k in MAPPING_TO_DATASET for k in mapping_keys)

    common_fields = {
        "embedding/model": __get_embedding_name(approach),
    }

    if not is_joint:
        lsr_benchmark.register_to_ir_datasets(dataset_id)
        dset = lsr_benchmark.load(dataset_id)
        if not dset.has_qrels():
            raise ValueError(f"The dataset {dataset_id} has no qrels.")
        ir_scores = {str(k): v for k, v in ir_measures.calc_aggregate(ir_measures_set, dset.qrels, run).items()}
        return [{
            **efficiency_results,
            **ir_scores,
            **common_fields,
            "tira-dataset-id": dataset_id,
            "ir-dataset-id": TIRA_DATASET_ID_TO_IR_DATASET_ID.get(dataset_id),
            "approach": approach,
        }]

    # --- Joint dataset handling ---
    datasets = [MAPPING_TO_DATASET[k] for k in mapping_keys]
    split_run = __split_run_by_dataset(run, mapping_keys)

    for dataset in datasets:
        lsr_benchmark.register_to_ir_datasets(dataset)

    # Per-dataset results (independent evaluation on each constituent dataset)
    per_dataset_results: "list[dict[str, Any]]" = []
    approach_stem = str(Path(approach)).split('/')[-1]

    for mapping_key, dataset in zip(mapping_keys, datasets):
        dset = lsr_benchmark.load(dataset)
        if not dset.has_qrels():
            raise ValueError(f"The dataset {dataset} has no qrels.")
        ir_scores = {str(k): v for k, v in ir_measures.calc_aggregate(
            ir_measures_set, dset.qrels, split_run[mapping_key]
        ).items()}
        per_dataset_results.append({
            **efficiency_results,
            **ir_scores,
            **common_fields,
            "tira-dataset-id": dataset,
            "ir-dataset-id": TIRA_DATASET_ID_TO_IR_DATASET_ID.get(dataset),
            "approach": f"{dataset}/{approach_stem}",
        })

    # Micro-average: evaluate the full joint run against the merged (prefixed) qrels
    mapped_qrels = __build_mapped_qrels(datasets)
    micro_ir_scores = {
        f"{name}(micro_avg)": v
        for name, v in (
            (str(k), v)
            for k, v in ir_measures.calc_aggregate(ir_measures_set, mapped_qrels, run).items()
        )
    }
    micro_result = {
        **efficiency_results,
        **micro_ir_scores,
        **common_fields,
        "tira-dataset-id": dataset_id,
        "ir-dataset-id": None,
        "approach": f"micro_avg/{approach_stem}",
    }

    # Macro-average: arithmetic mean of IR scores across the per-dataset results
    macro_ir_scores = __compute_macro_average(per_dataset_results, ir_measure_names)
    macro_result = {
        **efficiency_results,
        **macro_ir_scores,
        **common_fields,
        "tira-dataset-id": dataset_id,
        "ir-dataset-id": None,
        "approach": f"macro_avg/{approach_stem}",
    }

    return [*per_dataset_results, micro_result, macro_result]


@click.argument(
    "approaches",
    type=str,
    nargs=-1,
)
@click.option(
    "-m", "--measure",
    type=__parse_measure,
    required=False,
    multiple=True,
    default=["ndcg_cut.10", "nDCG(judged_only=True)@10", "P_10", "RR", "runtime_wallclock", "energy_total",
             "temperature_avg", "temperature_max"],
    help="The dataset id or a local directory.",
)
@click.option(
    "--upload",
    type=bool,
    default=False,
    is_flag=True,
    required=False,
    help="Upload to tira.",
)
@click.option(
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default="-",
    help="The output file to write to. Use - to print the results to stdout. Default: -",
)
def evaluate(approaches: "list[str]", measure: "list[tuple]", out: str, upload: bool) -> int:
    approaches = [x for xs in map(glob, approaches) for x in xs]
    output_routine = __get_output_routine(out)

    scores: "list[dict[str, Any]]" = []
    from tqdm import tqdm
    dataset_to_already_uploaded_approaches: "dict[str, set[str]]" = {}

    for approach in tqdm(approaches):
        approach_results = evaluate_approach(approach, measure)
        scores.extend(approach_results)

        if upload:
            from tira.tira_cli import upload_command
            from tira.rest_api_client import Client
            from lsr_benchmark.irds import TIRA_LSR_TASK_ID
            import time

            # Use the first result (per-dataset or single) for upload metadata
            first_result = approach_results[0]
            approach_name = (
                Path(approach).name
                + "-on-"
                + str(first_result["embedding/model"]).replace("/", "-")
            )
            metadata_of_run = yaml.safe_load(open(Path(approach) / "retrieval-metadata.yml"))
            team = metadata_of_run["actor"]["team"]
            dataset = metadata_of_run["data"]["test collection"]["name"]

            if "tiny-example" in dataset:
                continue

            if dataset not in dataset_to_already_uploaded_approaches:
                tira = Client()
                dataset_to_already_uploaded_approaches[dataset] = set(
                    tira.submissions(TIRA_LSR_TASK_ID, dataset)["software"].unique()
                )

            if approach_name in dataset_to_already_uploaded_approaches[dataset]:
                continue

            upload_command(
                dataset=dataset, directory=approach, dry_run=False,
                system=approach_name, tira_vm_id=team, default_task=TIRA_LSR_TASK_ID,
            )
            time.sleep(2)

    output_routine(pd.DataFrame(scores))
    return 0
