# neuralc
> A fast, lightweight deep learning library written in pure C — no dependencies, no bloat.

---

## What is neuralc

**neuralc** is an open-source neural network and tensor library written in C.  
It is **not** a copy or port of TensorFlow or PyTorch — it is an original library built from scratch, inspired by the same ideas but designed for:

- 🚀 **Raw speed** — C runs close to the metal, no Python overhead
- 🔩 **Simplicity** — readable source code anyone can understand and modify
- 📦 **Zero dependencies** — only needs a C compiler and `libm`
- 🖥️ **Portability** — runs on Linux, macOS, Windows, embedded systems

---

## Features (v0.1)

| Feature | Status |
|---|---|
| Multi-dimensional Tensors | ✅ Done |
| Element-wise math ops | ✅ Done |
| Matrix multiply | ✅ Done |
| Dense (fully-connected) layers | ✅ Done |
| ReLU, Sigmoid, Tanh, Softmax | ✅ Done |
| Backpropagation | ✅ Done |
| SGD + Momentum | ✅ Done |
| Adam optimizer | ✅ Done |
| MSE / BCE / Cross-Entropy loss | ✅ Done |
| Save & load weights | ✅ Done |
| Convolutional layers (CNN) | 🔜 Planned |
| Batch Normalization | 🔜 Planned |
| Dropout | 🔜 Planned |
| RNN / LSTM | 🔜 Planned |
| OpenMP multi-core support | 🔜 Planned |
| Python bindings (ctypes) | 🔜 Planned |
| GPU support via OpenCL | 🔜 Planned |

---

## Quick Start

### Requirements
- GCC or Clang (C11 or later)
- `libm` (standard math library, included on all platforms)

### Build & Run Demo

```bash
git clone https://github.com/Rohinthan/neuralc.git
cd neuralc
make
./demo
```

### Expected Output

```
=== Network Summary ===
  Layer 0: Dense(2 -> 8, ReLU)  params=24
  Layer 1: Dense(8 -> 1, Sigmoid)  params=9
  Total params: 33
=======================
Epoch    0  loss=0.535502
Epoch  500  loss=0.002831
Epoch 5000  loss=0.000015

--- XOR Predictions ---
  [0 XOR 0]  pred=0.0000  (expected 0)
  [0 XOR 1]  pred=1.0000  (expected 1)
  [1 XOR 0]  pred=1.0000  (expected 1)
  [1 XOR 1]  pred=0.0000  (expected 0)
```

---

## Usage Example

```c
#include "tensor.h"
#include "layer.h"
#include "nn.h"
#include "optimizer.h"

// Build a network
Network *net = nn_create();
nn_add_layer(net, dense_create(2, 8, ACT_RELU));
nn_add_layer(net, dense_create(8, 1, ACT_SIGMOID));

// Create optimizer
Adam *opt = adam_create(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);

// Training loop
for (int ep = 0; ep < 5000; ep++) {
    float loss = nn_train_step(net, X, Y, LOSS_BINARY_CROSS, pred, grad);
    adam_step(opt, net);
}

// Save weights
nn_save(net, "model.bin");

// Cleanup
nn_free(net);
adam_free(opt);
```

---

## File Structure

```
neuralc/
├── tensor.h / tensor.c       # Core tensor struct + all math ops
├── layer.h  / layer.c        # Dense layer (forward + backward)
├── nn.h     / nn.c           # Network, loss, backprop, save/load
├── optimizer.h / optimizer.c # SGD, Adam, LR schedulers
├── demo.c                    # XOR training demo
└── Makefile
```

---

## Roadmap

### v0.2 — Regularization & Data
- [ ] Dropout layer
- [ ] Batch Normalization
- [ ] CSV data loader

### v0.3 — CNN Support
- [ ] Conv2D layer
- [ ] MaxPool2D / AvgPool2D
- [ ] Flatten layer
- [ ] MNIST example

### v0.4 — Performance
- [ ] OpenMP multi-core parallelism
- [ ] BLAS integration (optional)
- [ ] Memory pool allocator

### v0.5 — Sequences
- [ ] RNN layer
- [ ] LSTM layer

### v1.0 — Bindings & GPU
- [ ] Python bindings via ctypes
- [ ] OpenCL GPU backend

---

## Contributing

We welcome all contributors! See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.  
Good first issues are labeled `good first issue` on GitHub.

---

## License

MIT License — free to use, modify, and distribute.

---

## Why not just use TensorFlow or PyTorch?

Those are incredible libraries — but they are massive, Python-first, and hard to embed.  
**neuralc** is for people who want to learn how neural networks *really* work at the C level,
or embed ML into firmware, games, or systems without Python.
