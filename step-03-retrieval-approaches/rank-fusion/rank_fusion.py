#!/usr/bin/env python3
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import yaml
from ranx import Run, fuse
from tirex_tracker import ExportFormat, register_metadata, tracking

RESOURCE_KEYS = (
    ("cpu", "energy used system"),
    ("gpu", "energy used system"),
    ("ram", "energy used system"),
    ("runtime", "wallclock"),
)


def write_joint_metadata(metadata: list[dict], out_path: Path) -> None:
    if not metadata:
        return

    combined_metadata = {
        "actor": metadata[0]["actor"],
        "data": metadata[0]["data"],
        "resources": {resource: {} for resource, _ in RESOURCE_KEYS},
    }

    for resource, key in RESOURCE_KEYS:
        total = 0.0
        unit = None
        all_null = True

        for item in metadata:
            value, current_unit = str(item["resources"][resource][key]).split(maxsplit=1)
            if unit is None:
                unit = current_unit
            elif current_unit != unit:
                raise ValueError(f"Mismatching units for resources.{resource}.{key}: {unit!r} and {current_unit!r}.")

            if value != "null":
                total += float(value)
                all_null = False

        combined_value = "null" if all_null else f"{total:g}"
        combined_metadata["resources"][resource][key] = f"{combined_value} {unit}"

    with out_path.open("w") as f:
        yaml.safe_dump(combined_metadata, f, sort_keys=False)


@click.command()
@click.argument("run-paths", type=click.Path(exists=True, resolve_path=True, path_type=Path), nargs=-1, required=True)
@click.option(
    "-n",
    "--norm",
    type=click.Choice(["min-max", "min-max-inverted", "max", "sum", "zmuv", "rank", "borda"]),
    default="min-max",
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["min", "med", "anz", "log_isr", "bordafuse", "condorcet", "max", "sum", "mnz", "lsr"]),
    default="min",
)
@click.option("--output", type=Path, required=True)
def main(run_paths, norm, method, output):
    metadata_files = ([], [])
    for run_path in run_paths:
        for i, meta_type in enumerate(["index", "retrieval"]):
            meta_path = run_path / f"{meta_type}-metadata.yml"
            if meta_path.exists():
                metadata_files[i].append(meta_path)

    metadata = ([], [])
    test_collection = None
    for i in range(2):
        for metadata_path in metadata_files[i]:
            with metadata_path.open() as f:
                meta = yaml.safe_load(f)
                metadata[i].append(meta)
                meta_collection = meta["data"]["test collection"]["name"]
                if test_collection is None:
                    test_collection = meta_collection
                elif test_collection != meta_collection:
                    raise ValueError(f"Mismatching test collection names. {test_collection!r} and {meta_collection!r}")

    output.mkdir(parents=True, exist_ok=True)
    register_metadata(
        {
            "actor": {"team": "reneuir-baselines"},
            "tag": f"rank-fusion-{norm}_norm-{method}_method",
            "data": metadata[0][0]["data"],
        }
    )

    runs = [Run.from_file(path / "run.txt.gz") for path in run_paths]
    retrieval_metadata_path = output / "retrieval-metadata.yml"
    with tracking(
        export_file_path=retrieval_metadata_path,
        export_format=ExportFormat.IR_METADATA,
    ):
        fused = fuse(runs=runs, norm="min-max", method="min")
    with open(retrieval_metadata_path) as f:
        metadata[1].append(yaml.safe_load(f))

    with TemporaryDirectory() as tmp:
        run_txt = Path(tmp) / "run.txt"
        fused.save(run_txt)
        with open(run_txt, "rb") as f_in, gzip.open(output / "run.txt.gz", "wb") as f_out:
            f_out.writelines(f_in)

    write_joint_metadata(metadata[0], output / "index-metadata.yml")
    write_joint_metadata(metadata[1], retrieval_metadata_path)
    # TODO: copy doc-ir-metadata.yml same for query


if __name__ == "__main__":
    main()
