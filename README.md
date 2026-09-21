# flash-engram介绍

简体中文|[English](./README_EN.md)

flash-engram 是面向 DeepSeek-V3 Engram 模块的鲲鹏亲和加速库，针对机头 CPU 的哈希计算与 Embedding 查表两大性能瓶颈进行深度优化。

## 最新消息

- [2026.09.30]：首次发布flash-engram v0.1.0-beta0版本，提供hash算子、float32和float16的embedding查表算子。

## 项目介绍

鲲鹏flash-engram能够为engram模块提供加速能力。在机头CPU上，Engram的性能瓶颈主要体现在两个关键环节：哈希计算与Embedding查表。
在哈希计算环节，每个token位置需要对多组N-gram组合进行乘法、异或、取余等运算。传统实现通常采用逐元素的串行标量运算，导致CPU算力被大量密集的细粒度运算所消耗。而在Embedding查表环节，由于多头多层结构的叠加，Embedding表的总体规模可达到数GB。同时，哈希值分布具有高度随机性，导致查表时的访问地址难以预测，CPU缓存命中率显著下降，Cache Miss频繁发生。每一次查表操作都不得不直接访问主内存，造成极大的访存延迟。
针对以上两大挑战，鲲鹏BoostKit团队推出了Engram加速解决方案，充分发挥鲲鹏处理器的硬件特性，从算力提升与访存优化两个维度，对机头CPU上的Engram推理进行了深度优化。本方案作为鲲鹏BoostKit推理加速解决方案的重要组成部分，现已在仓库中开放demo实现源码，供开发者参考与使用。

## 版本说明

|项目|说明|
|--|--|
|产品名称|鲲鹏BoostKit|
|产品版本|26.1.RC1|
|软件名称|flash-engram|
|软件版本|v0.1.0-beta0|

## 环境部署

**已验证硬件配置及软件配置**

|项目|说明|
|--|--|
|CPU类型|鲲鹏950处理器|
|编译器|GCC 12.3.1|
|python版本|3.11及以上|

## 目录结构

```text
├── LICENSE
├── README.md
├── requirements.txt          # 项目依赖
├── setup.py                  # 构建与安装脚本
├── example/
│   └── engram_demo_v1.py     # 使用示例（适配后）
└── flash_engram/             # Python 包源码
    ├── __init__.py
    ├── python_wrapper.py     # Python 接口封装
    └── core_engram.cpp       # C++ 核心实现
```

## 快速入门

1. 安装python依赖。

   ```shell
   pip install -r requirements.txt
   ```

2. 源码编译安装 flash_engram。

   ```shell
   pip install . --no-build-isolation
   ```

3. 执行适配flash_engram后的engram_demo_v1。

   ```shell
   cd example
   python engram_demo_v1.py
   ```

> ![说明](docs/figures/note.png) **说明**：若网络存在问题，可手动在huggingface官网下载deepseek-ai/DeepSeek-V3的tokenizer_config.json和tokenizer.json文件到本地目录/to/path/DeepSeek-V3，并指定tokenizer_name_or_path为/to/path/DeepSeek-V3

## 接口说明

flash_engram包提供的函数接口，出于最佳性能考虑，不对输入参数作校验，请在使用前确保输入参数符合要求。

|接口|说明|
|--|--|
|_get_ngram_hashes|seq_len必须大于0；input_ids、multipliers和head_vocab_sizes为连续内存|

## 贡献指南

如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交isssues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。

## 许可证书

许可证书为Mulan PSL v2，请参见 [《LICENSE》](./LICENSE) 文件。
