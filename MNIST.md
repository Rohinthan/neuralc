# MNIST Handwritten Digit Recognition with neuralc

This document explains how the MNIST demo works in neuralc, how to run it, and how contributors can extend it.

---

## What is MNIST?

MNIST is the most famous dataset in machine learning. It contains:
- **60,000 training images** of handwritten digits (0–9)
- **10,000 test images** for evaluation
- Each image is **28×28 pixels** in grayscale
- Written by humans in the 1990s, collected by Yann LeCun

It is the standard benchmark for testing if a neural network library works correctly. neuralc achieves **95.79% accuracy** on MNIST using C with zero external dependencies.

---

## Files Involved

```
neuralc/
├── mnist.h          — MNIST dataset loader header
├── mnist.c          — MNIST loader implementation
├── demo_mnist.c     — Training demo (main program)
└── mnist/
    ├── download.sh  — Script to download the dataset
    └── *.ubyte      — Dataset files (after download)
```

---

## Quick Start

### Step 1 — Clone the repo

```bash
git clone https://github.com/Rohinthan/neuralc.git
cd neuralc
```

### Step 2 — Build neuralc

```bash
make mnist_demo
```

You should see all `.c` files compiling and linking:
```
gcc -O2 -Wall -Wextra -std=c11 -c tensor.c -o tensor.o
gcc -O2 -Wall -Wextra -std=c11 -c layer.c -o layer.o
...
gcc -O2 -Wall -Wextra -std=c11 -o mnist_demo tensor.o layer.o ...
```

### Step 3 — Download the MNIST dataset

```bash
bash mnist/download.sh
```

This downloads ~60MB from public mirrors and extracts 4 files:

| File | Size | Contents |
|---|---|---|
| train-images-idx3-ubyte | 47MB | 60,000 training images |
| train-labels-idx1-ubyte | 58KB | 60,000 training labels |
| t10k-images-idx3-ubyte | 8MB | 10,000 test images |
| t10k-labels-idx1-ubyte | 10KB | 10,000 test labels |

### Step 4 — Run the demo

```bash
./mnist_demo
```

### Expected Output

```
╔══════════════════════════════════════════╗
║   neuralc — MNIST Digit Recognition      ║
╚══════════════════════════════════════════╝

Loading MNIST...
  Loaded in 0.22s
MNIST 'train': 60000 samples, 784 features, 10 classes
  Class distribution: 0:5923 1:6742 2:5958 ...

Sample digit:
Sample 0 — Label: 5
                        ░░▓▓▓▓  ▓▓████░░
            ░░▓▓▓▓████████████▓▓██████░░
        ████████████████████░░░░░░
        ...

Building network...
  Layer 0: Dense(784 -> 256, ReLU)   params=200960
  Layer 1: Dense(256 -> 128, ReLU)   params=32896
  Layer 2: Dense(128 -> 10, Softmax) params=1290
  Total params: 235146

  Epoch  Train Loss   Train Acc    Test Acc    Time
  ─────────────────────────────────────────────────
  1      0.4148       87.8%        94.0%       26s
  5      0.2371       93.4%        95.1%       32s
  10     0.2298       93.8%        94.8%       33s
  14     0.2296       93.7%        95.8%       33s
  20     0.2274       93.7%        95.2%       33s

  Best test accuracy: 95.79%
  Model saved → mnist_best.bin

  Sample Predictions:
  Index  Predicted  Actual
  0      7          7       ✓
  1      2          2       ✓
  2      1          1       ✓
  ...

✓ neuralc trained on 60,000 real images in pure C!
```

---

## How It Works — Step by Step

### 1. Loading the Data (`mnist.c`)

MNIST files use the IDX binary format. Each file starts with a header in big-endian 32-bit integers:

```
Images file:
  [magic=2051][n_images][rows=28][cols=28][pixel bytes...]

Labels file:
  [magic=2049][n_labels][label bytes...]
```

neuralc reads these and:
- Converts pixel values from `uint8 [0,255]` → `float32 [0.0, 1.0]`
- Flattens each 28×28 image → 784 float values
- One-hot encodes labels → 10-element vectors (e.g. digit 3 → [0,0,0,1,0,0,0,0,0,0])

### 2. Network Architecture

```
Input Layer       784 neurons  (one per pixel)
     ↓
Dense Layer 1     256 neurons  ReLU activation
     ↓
Dropout           30% drop rate (training only)
     ↓
Dense Layer 2     128 neurons  ReLU activation
     ↓
Dropout           30% drop rate (training only)
     ↓
Output Layer       10 neurons  Softmax activation
                  (one per digit class 0-9)
```

**Why this architecture?**
- 784 inputs because 28×28 = 784 pixels
- Two hidden layers give the network enough capacity to learn digit features
- Dropout prevents overfitting (memorizing training data)
- Softmax converts raw scores to probabilities that sum to 1.0

### 3. Training Loop

For each epoch:
1. **Shuffle** the training data (prevents learning order)
2. For each mini-batch of 64 images:
   - **Forward pass** — compute predictions
   - **Loss** — cross-entropy between predictions and labels
   - **Backward pass** — compute gradients through all layers
   - **Update** — Adam optimizer adjusts weights
3. **Evaluate** on the test set (dropout disabled)
4. **Save** if best accuracy so far

### 4. Loss Function — Cross-Entropy

For a single sample with true class `c`:
```
L = -log(predicted_probability_of_class_c)
```

If the network is confident and correct (probability = 0.99): `L = -log(0.99) ≈ 0.01` (low)
If the network is wrong (probability = 0.01): `L = -log(0.01) ≈ 4.6` (high)

**Important:** neuralc uses the **combined Softmax + Cross-Entropy gradient**:
```
gradient = (predicted - true_label) / batch_size
```
This is numerically stable and more efficient than computing them separately.

### 5. Adam Optimizer

Adam adapts the learning rate for each weight individually:
```
m = 0.9  * m + 0.1  * gradient          (momentum)
v = 0.999 * v + 0.001 * gradient²       (variance)
weight -= lr * m / (sqrt(v) + 1e-8)
```

Config used: `lr=0.001, beta1=0.9, beta2=0.999, weight_decay=1e-4`

---

## Configuration — How to Tune

At the top of `demo_mnist.c`:

```c
#define BATCH_SIZE  64    /* samples per gradient update     */
#define EPOCHS      20    /* number of full passes over data */
#define LR          0.001f /* learning rate                  */
#define DROP_RATE   0.3f  /* dropout probability             */
```

### Want faster training?
```c
#define BATCH_SIZE  128   /* larger batch = fewer updates = faster */
#define EPOCHS      10    /* fewer epochs = faster finish          */
```

### Want better accuracy?
```c
#define EPOCHS      50    /* more training                         */
#define LR          0.0005f /* slower, more careful learning       */
#define DROP_RATE   0.4f  /* more regularization                   */
```

### Want to test quickly?
Edit `mnist_load()` calls in `main()` to load fewer samples:
```c
MNISTData *train = mnist_load("...", "...", 5000);  /* only 5000 samples */
MNISTData *test  = mnist_load("...", "...", 1000);  /* only 1000 test   */
```

---

## Loading Your Own Saved Model

```c
#include "nn.h"
#include "layer.h"

Network *net = nn_create();
nn_add_layer(net, dense_create(784, 256, ACT_RELU));
nn_add_layer(net, dense_create(256, 128, ACT_RELU));
nn_add_layer(net, dense_create(128,  10, ACT_SOFTMAX));

// Load saved weights
nn_load(net, "mnist_best.bin");

// Run a prediction on a single image (784 floats, normalized [0,1])
int shX[2] = {1, 784};
int shY[2] = {1, 10};
Tensor *input  = tensor_zeros(shX, 2);
Tensor *output = tensor_zeros(shY, 2);

/* fill input->data with your 784 pixel values */

nn_forward(net, input, output);
int predicted_digit = tensor_argmax(output);
printf("Predicted: %d\n", predicted_digit);

tensor_free(input);
tensor_free(output);
nn_free(net);
```

---

## ASCII Art Digit Viewer

neuralc can display any MNIST digit in the terminal:

```c
#include "mnist.h"

MNISTData *data = mnist_load("mnist/train-images-idx3-ubyte",
                              "mnist/train-labels-idx1-ubyte", 0);
mnist_print_sample(data, 42);  // show sample at index 42
mnist_free(data);
```

Output:
```
Sample 42 — Label: 7
                ░░████████████████░░
              ██████████████████████
            ████░░░░░░░░░░░░░░████░░
            ...
```

---

## Contributor Ideas — What to Build Next

These are open tasks for contributors. Pick one and open a PR!

### Easy
- `mnist_print_batch()` — show a grid of multiple digits
- `mnist_find_errors()` — print all misclassified test samples
- `mnist_confusion_matrix()` — show which digits get confused with which
- Save/load dataset in CSV format

### Medium
- Train with Conv2D instead of Dense layers (should get ~99%)
- Add learning rate decay (reduce LR by 0.5 every 5 epochs)
- Add early stopping (stop when test accuracy stops improving)
- Training progress bar instead of epoch-by-epoch output

### Hard
- CIFAR-10 support (32×32 color images, 10 classes)
- Fashion-MNIST support (same format, harder dataset)
- t-SNE visualization of learned features
- Export predictions to CSV for analysis

---

## Results Summary

| Metric | Value |
|---|---|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Network parameters | 235,146 |
| Best test accuracy | **95.79%** |
| Training time per epoch | ~30 seconds (CPU) |
| Framework | Pure C, zero dependencies |
| Compiler | GCC with `-O2` |

---

## Troubleshooting

**`MNIST files not found`**
```bash
bash mnist/download.sh
```

**`make: mnist_demo: No such target`**
```bash
# Make sure you have the latest Makefile
make clean && make mnist_demo
```

**Loss stays at ~14 (not learning)**

This means the gradient is wrong. Make sure your `nn.c` has the correct cross-entropy gradient:
```c
/* CORRECT — combined softmax+CE gradient */
grad->data[b*classes + c] = (p - t) / (float)batch;

/* WRONG — causes exploding gradients */
grad->data[b*classes + c] = -(t / p) / (float)batch;
```

**Segmentation fault during training**

The intermediate tensors `h1d` and `h2d` must stay alive through the backward pass. Free them AFTER `dense_backward`, not before.

**`wget: command not found`**
```bash
sudo apt install wget
bash mnist/download.sh
```

---

## Platform Support

| Platform | Status |
|---|---|
| Ubuntu / Debian Linux | ✅ Tested |
| macOS | ✅ Should work |
| Windows (WSL) | ✅ Should work |
| Raspberry Pi | ✅ Should work (slower) |
| Embedded (no OS) | ⚠️ Needs porting (no malloc) |

---

*neuralc — Deep learning in pure. No dependencies. No limits.*
