# LSP

[LSP](https://github.com/thefxperson/hierarchical_pruning) is a learned sparse retrieval engine written in Rust that
extends BMP (SIGIR '24) and SP (SIGIR '25') with a hierarchical superblock index:
document embeddings are quantized to 8-bit impacts and laid out in fixed-size blocks,
and queries are processed by pruning (super)blocks whose upper-bound score cannot enter the top-k.

The entrypoint indexes the document embeddings in memory and answers all queries in a single batched
call into the Rust engine. Currently only `linux/amd64` is supported 

Tunable knobs (defaults are rank-safe): 
`--gamma` (guaranteed number of superblocks), 
`--mu` (threshold overestimation),
`--eta` (probabalistic safeness (and within-block threshold overestimation)),
`--beta` (query pruning).
Indexing parameters:
`--bsize` (superblock,block sizes),
`--compression` (block upper-bound storage: `simdbp`, `superblock` or `raw`),
`--quant` (quantization of those upper bounds; they are stored as `impact >> quant`, so the
default of `4` keeps them in 4 bits, which is what makes the SIMD-BP packing pay off, and `8`
keeps the full 8-bit impact),
`--compress-doc` (document index layout: `seismic`, `flatinv` or `bmp` — all three rank
identically and differ only in memory footprint and block-scoring speed),
`--threads` (parallel search).

`--k` must be 10, 100 or 1000: LSP only keeps kth-score statistics at those depths.

## Submission

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
    --mount-directory '$embeddings=lsr-benchmark/lightning-ir/naver-splade-v3-doc' \
    --dry-run
```

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt` (this compiles LSP from source and needs the Rust nightly toolchain via [rustup](https://rustup.rs/)).

To make predictions on a dataset, run:

```
./build-and-search-lsp-index.py --dataset lsr-benchmark/clueweb09/en/trec-web-2009 --embedding naver/splade-v3 --output output-dir
```
