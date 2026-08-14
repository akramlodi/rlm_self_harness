
---

<h1 align="center" style="font-size:2.8em">
<span>Recursive Language Models (<span style="color:orange">RLM</span>s)</span>
</h1>

<p align="center" style="font-size:1.3em">
  <a href="https://arxiv.org/abs/2512.24601">Full Paper</a> •
  <a href="https://alexzhang13.github.io/blog/2025/rlm/">Blogpost</a> •
  <a href="https://alexzhang13.github.io/rlm/">Documentation</a> •
  <a href="https://github.com/alexzhang13/rlm-minimal">RLM Minimal</a>
</p>

<p align="center">
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/style.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/test.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/test.yml/badge.svg" alt="Test" />
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.24601">
    <img src="media/paper_preview.png" alt="Paper Preview" width="300"/>
  </a>
</p>

## Overview
Recursive Language Models (RLMs) are a task-agnostic inference paradigm for language models (LMs) to handle near-infinite length contexts by enabling the LM to *programmatically* examine, decompose, and recursively call itself over its input. RLMs replace the canonical `llm.completion(prompt, model)` call with a `rlm.completion(prompt, model)` call, acting as a "language model". RLMs offload the context as a variable in a REPL environment that the LM can interact with and launch sub-LM calls inside of.

RLMs are a bet on future "language model" design choices. We argue for a [CodeAct](https://arxiv.org/abs/2402.01030)-style harness (i.e. all language models should have access to a code environment) with sub-(R)LM calls as functions in code, and context / prompts as objects in code. RLMs explicitly defer code execution with sub-calls as functions to the language model itself, which is incredibly flexible and lends itself well to scale if trained correctly. We want to move away from the JSON tool-calling standard for both sub-agents and generic tool calls. The naming comes from the fact that such a system is itself a "language model" (a probabilistic mapping from text to text) that builds around and relies on recursive sub-LLM calls.

This repository provides both an extensible inference engine and training environment for using RLMs around standard API-based and local LLMs. The initial experiments and idea were proposed in a [blogpost](https://alexzhang13.github.io/blog/2025/rlm/) in 2025, with expanded results in an [arXiv preprint](https://arxiv.org/abs/2512.24601).

We now also include a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) training environment based on Prime Intellect's [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) in the `training/` folder. Train your own RLMs, which directly can be plugged into our inference engine!

> [!NOTE]
> This repository contains inference code for RLMs with support for various sandbox environments. Open-source contributions are welcome. This repository is maintained by the authors of the paper from the MIT OASYS lab.

## Quick Setup
> [!NOTE]
> `rlms` requires **Python 3.11 or later**.

You can try out RLMs quickly by installing from PyPi:
```bash
pip install rlms
```

The default RLM client uses a REPL environment that runs on the host process through Python `exec` calls. It uses the same virtual environment as the host process (i.e. it will have access to the same dependencies), but with some limitations in its available global modules. As an example, we can call RLM completions using GPT-5-nano:
```python
from rlm import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-nano"},
    verbose=True,  # For printing to console with rich, disabled by default.
