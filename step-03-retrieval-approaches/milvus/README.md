# Milvus Retrieval

Sparse inner-product retrieval with
[Milvus](https://github.com/milvus-io/milvus). The runtime image is based on
the official Milvus server image and starts an isolated standalone server with
embedded etcd and local storage for each benchmark run.

Documents are indexed with Milvus's native `SPARSE_INVERTED_INDEX`. The
original benchmark document IDs are stored in a `VARCHAR` field while numeric
internal IDs are used as primary keys.

## Development

Open this directory in a Dev Container, or build and run it directly:

```bash
docker build \
    -f .devcontainer/Dockerfile \
    -t lsr-benchmark-milvus-dev \
    .
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/workspace" \
    -w /workspace \
    lsr-benchmark-milvus-dev \
    sh -c 'pytest -v && ruff check .'
```

## Usage

```bash
python3 milvus_retrieval.py \
    --dataset <dataset> \
    --embedding <embedding> \
    --output <output-dir> \
    --k 1000
```

`--algorithm` selects `DAAT_MAXSCORE`, `DAAT_WAND`, or `TAAT_NAIVE`.
`--index-batch-size` and `--query-batch-size` control ingestion and search
batching. `--drop-ratio-build` and `--drop-ratio-search` can prune low-weight
document and query dimensions; both default to `0`, which preserves exact
inner-product scoring.

## Submission

```bash
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --dataset tiny-example-20251002_0-training \
    --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
    --mount-directory '$embeddings=lsr-benchmark/lightning-ir/naver-splade-v3-doc' \
    --platform host \
    --dry-run
```
