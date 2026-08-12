# Tiny Transformer Lab

A small research-oriented Transformer implementation built from scratch with NumPy and CuPy.

The goal of this project is to understand and experiment with the internal mechanics of Transformer models without relying on PyTorch autograd or high-level neural network layers.

This repository is a cleaned-up public version of a larger experimental codebase that I use to study training dynamics, model internals, and small language model architectures.

## Why this project exists

Modern deep learning frameworks hide a large part of the mechanics behind automatic differentiation and ready-made layers.

In this project, core neural network components are implemented manually in order to explore:

- forward and backward propagation;
- gradient accumulation;
- attention-related building blocks;
- positional encoding with RoPE;
- CPU/GPU execution with the same model code;
- training dynamics of small Transformer models;
- reproducible numerical experiments.

The project is primarily intended for research, experimentation, and learning rather than production use.

## Current implementation

The public repository currently contains the foundational components of the model.

### Linear layer

A fully connected layer implemented without autograd:

```text
Y = X @ W.T + b
