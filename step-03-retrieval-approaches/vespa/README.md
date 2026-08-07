# Vespa Retrieval

Sparse maximum-inner-product retrieval with
[Vespa](https://github.com/vespa-engine/vespa). The runtime image starts a
single-node Vespa deployment and uses the native `wand` query operator over a
fast-search `weightedset<int>` attribute.

Vespa weighted-set keys and weights are signed 32-bit integers. Benchmark
token IDs are used directly as keys. Document values are globally linearly
quantized, while each query is linearly quantized independently. Both scales
are removed from the returned scores. Vespa guarantees the exact top-k result
for the resulting quantized sparse vectors.

## Development

Open this directory in a Dev Container, or build and run it directly:

```bash
docker build \
    -f .devcontainer/Dockerfile \
    -t lsr-benchmark-vespa-dev \
    .
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/workspace" \
    -w /workspace \
    lsr-benchmark-vespa-dev \
    sh -c 'pytest -v && ruff check .'
```

## Usage

```bash
python3 vespa_retrieval.py \
    --dataset <dataset> \
    --embedding <embedding> \
    --output <output-dir> \
    --k 1000
```

`--max-weight` controls the integer quantization range.
`--feed-workers` controls concurrent document ingestion.

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
