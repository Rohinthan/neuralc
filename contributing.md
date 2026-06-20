# Contributing to neuralc

Thank you for your interest in contributing! neuralc is a community-built C deep learning library and every contribution matters — from fixing a typo to implementing a full CNN layer.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Style](#coding-style)
- [Good First Issues](#good-first-issues)
- [Roadmap Tasks](#roadmap-tasks)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/Rohinthan/neuralc.git
cd neuralc
```

### 2. Build and verify everything works

```bash
make
./demo
```

You should see XOR training converge to near-zero loss and perfect predictions.

### 3. Create a branch for your work

```bash
git checkout -b feature/dropout-layer
```

---

## Project Structure

```
neuralc/
├── tensor.h / tensor.c       # Tensor struct, memory, math ops
├── layer.h  / layer.c        # Dense layer forward + backward
├── nn.h     / nn.c           # Network container, loss, backprop
├── optimizer.h / optimizer.c # SGD, Adam, LR schedulers
├── demo.c                    # Example: XOR training
└── Makefile
```

Each module is self-contained. If you're adding a new layer type (e.g. Conv2D), create `conv.h` and `conv.c` following the same pattern as `layer.h / layer.c`.

---

## How to Contribute

### Bug fixes
- Open an issue first describing the bug and how to reproduce it
- Reference the issue number in your PR

### New features
- Check the Roadmap in README.md first
- Open an issue to discuss the design before writing code
- Keep new files focused — one layer type or feature per file

### Documentation
- Improve comments in headers
- Add examples to the README
- Write a new demo (e.g. `demo_mnist.c`)

### Performance
- Benchmark before and after your change
- Note the platform (CPU, OS, compiler) in your PR

---

## Coding Style

neuralc follows a simple, consistent C style:

```c
/* Use C-style block comments, not // */

/* Function names: module_verb_noun */
void tensor_add(const Tensor *a, const Tensor *b, Tensor *out);
void dense_forward(DenseLayer *l, const Tensor *input, Tensor *output);

/* Structs use PascalCase */
typedef struct { ... } DenseLayer;

/* Constants and enums use ALL_CAPS */
#define TENSOR_MAX_DIMS 8
typedef enum { ACT_RELU = 1 } Activation;

/* Always check malloc return values */
Tensor *t = malloc(sizeof(Tensor));
if (!t) return NULL;

/* Free what you allocate */
void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->owns_data) free(t->data);
    free(t);
}
```

Rules:
- C11 standard (`-std=c11`)
- 4-space indentation, no tabs
- Max ~100 chars per line
- No external dependencies (only `<stdlib.h>`, `<math.h>`, `<string.h>`, `<stdio.h>`)
- Compile clean with `-Wall -Wextra` (warnings are treated seriously)

---

## Good First Issues

These are great starting points if you're new to the project:

| Task | Difficulty | File(s) |
|---|---|---|
| Add `tensor_sum_axis()` — sum along one axis | Easy | tensor.h / tensor.c |
| Add `tensor_argmax()` — index of max value | Easy | tensor.h / tensor.c |
| Add accuracy metric utility function | Easy | nn.h / nn.c |
| Write a CSV data loader | Medium | dataloader.h / dataloader.c |
| Implement Dropout layer | Medium | layer.h / layer.c |
| Implement Batch Normalization | Medium | layer.h / layer.c |
| Add RMSProp optimizer | Medium | optimizer.h / optimizer.c |
| Write MNIST demo (load IDX format) | Medium | demo_mnist.c |
| Implement Conv2D forward pass | Hard | conv.h / conv.c |
| Add OpenMP parallelism to matmul | Hard | tensor.c |

To claim an issue, comment on it on GitHub so we know you're working on it.

---

## Roadmap Tasks (bigger contributions)

If you want to take on a major feature:

### Conv2D Layer
- Files: `conv.h`, `conv.c`
- Must implement: `conv2d_forward()`, `conv2d_backward()`
- Follow the same struct pattern as `DenseLayer`

### RNN / LSTM
- Files: `rnn.h`, `rnn.c`
- Must handle variable-length sequences
- Include a simple text prediction demo

### OpenCL GPU Backend
- Files: `gpu/tensor_gpu.h`, `gpu/tensor_gpu.c`
- Must be optional — CPU path must still work without OpenCL

### Python Bindings
- Files: `python/cforge.py`
- Use `ctypes` to wrap the C API
- Include a Python training example

---

## Pull Request Process

1. Make sure `make` succeeds with no errors
2. Make sure `./demo` runs and XOR converges correctly
3. Add or update comments in any header you modify
4. Keep PRs focused — one feature or fix per PR
5. Write a clear PR description:
   - What does this add/fix?
   - How was it tested?
   - Any known limitations?

A maintainer will review within a few days. We may suggest changes — that's normal and not a rejection.

---

## Code of Conduct

- Be respectful and constructive
- Welcome contributors of all experience levels
- No harassment, gatekeeping, or dismissiveness
- If you see bad behavior, open an issue or email the maintainer

We want neuralc to be a place where people learning C and ML feel welcome.

---

## Questions?

Open a GitHub Discussion or issue and we'll help you get started.  
Thank you for helping build neuralc
