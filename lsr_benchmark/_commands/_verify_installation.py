import os
from tira.rest_api_client import Client
from tira.check_format import _fmt, check_format, log_message
from tira.io_utils import docker_supported_target_platform
from pathlib import Path

from tempfile import TemporaryDirectory

# TODO: Pull this from archive.tira.io
EXAMPLE_RETRIEVAL_ENGINE = {
    "linux/amd64": {
        "naive-search": {
            "image": "ghcr.io/reneuir/lsr-benchmark/naive-search:amd64-4b508-a8fba",
            "command": "/build-and-search-naive-index.py --dataset $inputDataset --use-u32 true --embedding $embeddings --output $outputDir"
        },
        "seismic": {
            "image": "ghcr.io/reneuir/lsr-benchmark/seismic:amd64-54d65-202d6",
            "command": "/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir",
        },
        "pyterrier-splade": {
            "image": "ghcr.io/reneuir/lsr-benchmark/pyterrier-splade:amd64-54d65-f87d4",
            "command": "/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir",
        },
        "duckdb": {
            "image": "ghcr.io/reneuir/lsr-benchmark/duckdb:amd64-54d65-1a71b",
            "command": "/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir",
        },
        
        "numpy-exhaustive": {
            "image": "ghcr.io/reneuir/lsr-benchmark/numpy-exhaustive:amd64-bc9aa-9beac",
            "command": "/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir",
        },
    },
    "linux/arm64": {
        "naive-search": {
            "image": "ghcr.io/reneuir/lsr-benchmark/naive-search:arm64-4b508-f7db4",
            "command": "/build-and-search-naive-index.py --dataset $inputDataset --use-u32 true --embedding $embeddings --output $outputDir"
        },
        "seismic": {
            "image": "ghcr.io/reneuir/lsr-benchmark/seismic:arm64-54d65-ebb95",
            "command": "/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir",
        },
        "pyterrier-splade": {
            "image": "ghcr.io/reneuir/lsr-benchmark/pyterrier-splade:arm64-54d65-ab053",
            "command": "/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir",
        },
        "duckdb": {
            "image": "ghcr.io/reneuir/lsr-benchmark/duckdb:arm64-54d65-a367d",
            "command": "/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir",
        },

        "numpy-exhaustive": {
            "image": "ghcr.io/reneuir/lsr-benchmark/numpy-exhaustive:arm64-bc9aa-00322",
            "command": "/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir",
        },
    }
}

def get_spot_check_embeddings(client):
    print("Download some spot-check embeddings...")
    
    client.download_dataset("lsr-benchmark", "tiny-example-20251002_0-training", False)
    client.download_dataset("lsr-benchmark", "tiny-example-20251002_0-training", True)
    return client.get_run_output("lsr-benchmark/lightning-ir/naver-splade-v3-doc", "tiny-example-20251002_0-training")

def verify_docker_installation(client, image):
    print("Test docker/podman installation")
    if not client.local_execution.docker_is_installed_failsave():
        raise ValueError("Docker/Podman is not installed.")
    
    print("Pull an example retrieval engine for spot-checks.")
    client.local_execution.ensure_image_available_locally(image)

def verify_installation() -> int:
    all_messages = []

    def print_message(message, level):
        all_messages.append((message, level))
        os.system("cls" if os.name == "nt" else "clear") # noqa: S605
        print("lsr-benchmark verify-installation")
        for msg, lvl in all_messages:
            log_message(msg, lvl)

    client = Client()
    platform = docker_supported_target_platform()

    if platform not in ("linux/amd64", "linux/arm64"):
        log_message(f"The platform {docker_supported_target_platform()} is not supported.", _fmt.ERROR)

    engine = EXAMPLE_RETRIEVAL_ENGINE[platform]["naive-search"]
    print_message(f"The platform {docker_supported_target_platform()} is supported.", _fmt.OK)

    embeddings = get_spot_check_embeddings(client)
    print_message("Spot-Check embeddings are downloaded.", _fmt.OK)

    verify_docker_installation(client, engine["image"])
    
    print_message("Docker/Podman is working.", _fmt.OK)
    print_message("An example retrieval engine was pulled.", _fmt.OK)

    print("Run example retrieval engine")
    out_dir = TemporaryDirectory().name
    input_dir = client.download_dataset("lsr-benchmark", "tiny-example-20251002_0-training", False)
    client.local_execution.run(None, engine["image"], engine["command"], input_dir=input_dir, output_dir=out_dir, mount_directory={"embeddings": embeddings}, platform=platform)

    status, msg = check_format(Path(out_dir), "run.txt")
    if status != _fmt.OK:
        print(msg)
        return 1
    else:
        print_message("The example retrieval engine produced valid results.", _fmt.OK)
        return 0
