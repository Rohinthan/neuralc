# neuralc

A fast, lightweight deep learning library written in pure C — no dependencies, no bloat.

---

## Overview

**neuralc** is an open-source neural network and tensor computation library built entirely in C.

It is not a wrapper, port, or binding of an existing framework like PyTorch or TensorFlow. It is a from-scratch implementation focused on performance, simplicity, and full control over low-level execution.

## Motivation

Most modern ML frameworks are Python-dependent, large, complex, and difficult to embed in low-level systems. neuralc explores a different approach: build a lightweight, fully controllable deep learning engine in C that exposes how neural networks actually work under the hood.

What started as an experiment has evolved into a fully functional ML runtime capable of training real models on real datasets.

## Design Goals

| Goal | Description |
|---|---|
| **Performance-first** | Runs close to the metal with minimal overhead |
| **Simplicity** | Clean, readable C code |
| **Zero dependencies** | Requires only a C compiler and `libm` |
| **Portability** | Runs on Linux, macOS, Windows, and embedded systems |
| **Modularity** | Configurable, `menuconfig`-style build system |

## Proven Results — MNIST (95.79% Accuracy)

Trained on 60,000 handwritten digit images using only C:

```

<img width="707" height="512" alt="image" src="https://github.com/user-attachments/assets/e82b1902-22d9-47f2-8fc9-f5a0ba2f594f" />

```


See [`MNIST.md`](./MNIST.md) for full details.

## Features

**Core Engine**
- Multi-dimensional tensor system
- Element-wise operations
- Matrix multiplication

**Neural Networks**
- Dense (fully connected) layers
- CNN (Conv2D, MaxPool, Flatten)
- RNN & LSTM (with BPTT)

**Activations**
- ReLU, Sigmoid, Tanh, Softmax

**Training**
- Backpropagation
- Gradient clipping
- Loss functions: MSE, BCE, Cross-Entropy

**Optimizers**
- SGD with Momentum
- Adam
- RMSProp

**Data & Utilities**
- CSV data loader
- MNIST dataset support
- Binary model save/load

**Performance**
- OpenMP multi-core support
- Python bindings via `ctypes`
- OpenCL GPU backend
- CUDA backend *(in progress)*

## Build & Run

### Requirements

- GCC or Clang (C11+)
- `libm`

### Build

```bash
git clone https://github.com/Rohinthan/neuralc.git
cd neuralc
make
```

### Run Examples

```bash
./demo        # XOR demo
./neuralc     # full feature demo
```

### MNIST Demo

```bash
bash mnist/download.sh
make mnist_demo
./mnist_demo
```

### RNN / LSTM Demo

```bash
make rnn_demo
./rnn_demo
```

## Python Bindings

```bash
make libneuralc

python3 -m venv venv
source venv/bin/activate
pip install numpy

cd python
python3 neuralc.py
```

This will:
- Load the shared library (`libneuralc.so`)
- Run training from Python
- Confirm backend availability

## Configuration System

neuralc includes a `menuconfig`-style build system:

```bash
make config
```

This generates `neuralc_config.h`, used for compile-time control:

```c
#ifdef USE_CUDA
    // GPU backend
#else
    // CPU fallback
#endif
```

**Benefits:**
- Portable across systems
- GPU/CPU switching
- Debug and performance tuning
- Clean, modular builds

## Project Structure

```
neuralc/
├── include/        # headers
├── src/            # core implementation
├── config/         # configuration system
├── examples/       # demos
├── tests/          # test cases
├── python/         # Python bindings
├── mnist/          # dataset tools
├── build/          # compiled objects
├── Makefile
└── neuralc_config.h
```

## Build Targets

| Target | Description |
|---|---|
| `make` | Build all |
| `make config` | Configure features |
| `make mnist_demo` | MNIST training |
| `make rnn_demo` | Sequence models |
| `make omp` | OpenMP support |
| `make gpu` | OpenCL backend |
| `make libneuralc` | Shared library |
| `make clean` | Clean build artifacts |

## GPU Development

CUDA backend support is currently under active development:

- `cuda_backend.cu`
- `cuda_backend.h`

Designed for:
- Custom CUDA kernels
- Clean backend abstraction (CPU ↔ GPU)
- Future performance optimization

## Roadmap

- [x] Core tensor engine
- [x] CNN / RNN / LSTM support
- [x] OpenMP parallelism
- [x] Python bindings
- [x] OpenCL GPU backend
- [ ] CUDA kernel optimization
- [ ] Advanced autograd improvements
- [ ] Model export formats

## Contributing

Contributions are welcome!

1. Check [`CONTRIBUTING.md`](./CONTRIBUTING.md)
2. Look for issues tagged `good first issue`
3. Help improve performance, features, and GPU support

## Why neuralc?

Use neuralc if you want to:
- Learn how deep learning works internally
- Build ML systems without Python
- Run neural networks in low-level environments
- Experiment with custom backends (CPU/GPU)

## License

Licensed under the [MIT License](./LICENSE) — free to use and modify.

## Author

Built with curiosity and persistence — starting from scratch, evolving into a complete deep learning runtime.
