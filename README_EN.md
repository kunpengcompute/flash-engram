# flash-engram Introduction

English|[简体中文](./README.md)

flash-engram is a Kunpeng-optimized acceleration library for the DeepSeek-V3 Engram module. It targets the two main performance bottlenecks of the head-CPU stage: hash computation and Embedding table lookup.

## What's New

- [2026.09.30]: First release of flash-engram v0.1.0-beta0. Provides hash operator and float32 / float16 embedding lookup operators.

## Overview

flash-engram accelerates the Engram module on Kunpeng CPUs. On the head CPU, the Engram performance bottleneck shows up in two main places: hash computation and Embedding lookup.
For hash computation, every token position has to run a chain of multiplies, XORs and modulo operations across multiple N-gram combinations. Traditional implementations run these as element-wise serial scalar operations, which consumes CPU compute on a huge amount of fine-grained work. For Embedding lookup, the multi-head, multi-layer structure means the total Embedding table reaches several GB. The hash values are highly random, so lookup addresses are hard to predict, the CPU cache hit rate drops significantly, and cache misses happen frequently. Every lookup ends up hitting main memory, producing very large access latency.
For these two challenges, the Kunpeng BoostKit team built the Engram acceleration solution. It leverages Kunpeng's hardware features and deeply optimizes the head-CPU Engram inference along two dimensions: compute throughput and memory-access efficiency. As an important part of the Kunpeng BoostKit inference acceleration solution, this project now publishes the demo source code in this repository as a reference for developers.

## Version

|Item|Description|
|--|--|
|Product Name|Kunpeng BoostKit|
|Product Version|26.1.RC1|
|Software Name|flash-engram|
|Software Version|v0.1.0-beta0|

## Environment

**Verified hardware and software configuration**

|Item|Description|
|--|--|
|CPU|Kunpeng 950|
|Compiler|GCC 12.3.1|
|Python|3.11 or later|

## Directory Structure

```text
├── LICENSE
├── README.md
├── requirements.txt          # project dependencies
├── setup.py                  # build & install script
├── example/
│   └── engram_demo_v1.py     # usage example (adapted)
└── flash_engram/             # Python package source
    ├── __init__.py
    ├── python_wrapper.py     # Python wrapper
    └── core_engram.cpp       # C++ core implementation
```

## Quick Start

1. Install Python dependencies.

   ```shell
   pip install -r requirements.txt
   ```

2. Build and install flash_engram from source.

   ```shell
   pip install . --no-build-isolation
   ```

3. Run the adapted `engram_demo_v1`.

   ```shell
   cd example
   python engram_demo_v1.py
   ```

> ![](./docs/zh/public_sys-resources/icon-note.gif) **Note:** If network access is unavailable, manually download the `tokenizer_config.json` and `tokenizer.json` files of `deepseek-ai/DeepSeek-V3` from the Hugging Face website into the local directory `/to/path/DeepSeek-V3`, and set `tokenizer_name_or_path` to `/to/path/DeepSeek-V3`.

## Interface Description

The interfaces exposed by the `flash_engram` package skip input validation for the best performance. Make sure your inputs meet the requirements before calling.

|Interface|Description|
|--|--|
|_get_ngram_hashes|`seq_len` must be greater than 0; `input_ids`, `multipliers`, and `head_vocab_sizes` must be contiguous in memory|

## Contributing

If you run into any issues, or want to request a feature or report a bug, please open an issue. See the [contribution guide](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md) for more details.

## License

This project is licensed under Mulan PSL v2. See the [《LICENSE》](./LICENSE) file for details.
