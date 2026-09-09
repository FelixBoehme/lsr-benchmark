# Embedding Engines in the lsr_benchmark

This directory contains the embedding engines that we currently have in the lsr_benchmark. We aim to organize the lsr_benchmark as mono-repo that is fully self contained with simple and clean implementations, for that reason, if you want to contribute new embedding engines (we would be very happy about that), please make a pull request.

Please run `lsr-benchmark overview` for an up-to-date overview over all datasets and all embeddings.

Currently, we have 6 embedding engines (produce more embeddings, as one engine often can handle different models) that can run lsr embeddings:

- [bge-m3](bge-m3)
- [lexical](lexical)
- [lexical-index](lexical-index)
- [lightning-ir](lightning-ir)
- [mteb](mteb)
- [sentence-transformers](sentence-transformers)

## Remaining Embedding Engines

We are in the progress of adding the following remaining embedding engines:

- [ ] Deep-CT: Maik
- [ ] DocT5Query: Maik
- [ ] Different bi-encoder, especially based on LLMs

