# Qdrant Retrieval

Exact sparse inner-product retrieval with
[Qdrant](https://github.com/qdrant/qdrant). The runtime image bundles a Qdrant
server and uses its native float32 sparse inverted index. Documents receive
numeric internal point IDs while their original benchmark IDs are preserved in
the payload.

The command starts an isolated Qdrant server for each run, builds a temporary
collection, executes exact batched sparse queries, and removes the collection
storage when the run finishes.

## Development

Open this directory in a Dev Container, or build and run it directly:

```bash
docker build \
    -f .devcontainer/Dockerfile \
    -t lsr-benchmark-qdrant-dev \
    .
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/workspace" \
    -w /workspace \
    lsr-benchmark-qdrant-dev \
    sh -c 'pytest -v && ruff check .'
```

## Usage

```bash
python qdrant_retrieval.py \
    --dataset <dataset> \
    --embedding <embedding> \
    --output <output-dir> \
    --k 1000
```

`--index-batch-size` controls the number of documents per upsert and
`--query-batch-size` controls the number of searches per batch request. The
sparse index is held in memory by default; use `--on-disk` to store it on disk.
Both modes use float32 weights, no IDF modifier, and exact search.

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
