# PyTerrier Naive Baseline for LSR

This is a naive baseline for the lsr-benchmark that aims to fulfull the input/output contract while actually not doing any LSR at all. The idea is that this can be used as a baseline that has no dependencies to embeddings to test pipelines without much dependencies.


## Submission

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/run-pyterrier.py --dataset $inputDataset --output $outputDir --index $embeddings --retrieval BM25' \
    --dry-run
```

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To make predictions on a dataset, run:

```
./run-pyterrier.py --dataset clueweb09/en/trec-web-2009 --index /path/to/index --retrieval BM25 --output output-dir
```
