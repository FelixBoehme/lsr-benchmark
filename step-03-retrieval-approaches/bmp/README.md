# BMP Retrieval

Learned sparse retrieval with
[BMP](https://github.com/pisa-engine/BMP), the Block-Max Pruning engine
published at SIGIR 2024.

The adapter globally quantizes positive document impacts to 8 bits, as required
by BMP, builds the block-based inverted and forward indexes, and uses BMP's
Python `Searcher` for query processing. BMP internally quantizes each query
relative to its largest term weight. Because BMP accumulates scores in 16 bits,
the adapter automatically lowers the configured maximum document impact when
the longest benchmark query could otherwise overflow the accumulator.

The pinned BMP revision omits the final candidate block during query
processing and uses an inclusive kth-score estimate as an exclusive heap
threshold. [`bmp-final-block.patch`](bmp-final-block.patch) corrects both
issues in the development and runtime builds, which is especially important
for small corpora, large block sizes, and tied scores.

The rank-safe defaults retain all query terms (`--beta 1`) and use BMP's safe
block-pruning threshold (`--alpha 1`). Lower values trade ranking accuracy for
retrieval speed.

## Options

- `--block-size`: number of documents per BMP block, from `1` to `256`;
  defaults to `8`. BMP stores in-block document offsets in 8 bits.
- `--compress-range/--no-compress-range`: controls compression of block range
  maximum scores.
- `--max-document-impact`: maximum globally quantized document impact, from
  `1` to `255`.
- `--alpha`: BMP block-pruning aggressiveness.
- `--beta`: fraction of the highest-weight query terms retained.

BMP stores internal term identifiers in 16 bits. The adapter rejects document
embeddings with more than 65,536 unique dimensions instead of allowing term
identifier collisions.

## Architecture Support

The container builds BMP from its pinned source revision instead of using its
x86-64-only PyPI wheels. The source implementation supports both AMD64 and
ARM64.

## Development

Build the development container and run the tests and linter:

```bash
docker build \
    -f .devcontainer/Dockerfile \
    -t lsr-benchmark-bmp-dev \
    .

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/workspace" \
    -w /workspace \
    lsr-benchmark-bmp-dev \
    sh -c 'pytest -v && ruff check .'
```

Run BMP on a benchmark dataset:

```bash
python bmp_retrieval.py \
    --dataset lsr-benchmark/clueweb09/en/trec-web-2009 \
    --embedding naver/splade-v3 \
    --output output-dir
```

## Submission

```bash
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
    --mount-directory '$embeddings=lsr-benchmark/lightning-ir/naver-splade-v3-doc' \
    --platform host \
    --dry-run
```
