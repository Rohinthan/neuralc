# neuralc Configuration UI

neuralc includes a **kernel-style interactive configuration UI** — inspire by Linux `menuconfig` — that lets you configure every aspect of the library without editing any C code.

---

## Quick Start

```bash
make config
./neuralc_config
```

Then rebuild with your new settings:

```bash
make clean && make
```

---

## What It Looks Like

```
╔══════════════════════════════════════════════════════════════╗
║  neuralc Configuration                        neuralc v0.1  ║
╠══════════════════════════════════════════════════════════════╣
║ ► Main Menu                                                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Arrow keys navigate. Space/Enter toggle. S save. Q quit.  ║
║   ───────────────────────────────────────────────────────    ║
║   Performance Settings   --->                                ║
║   Training Defaults      --->                                ║
║   Memory Settings        --->                                ║
║   Debug & Profiling      --->                                ║
║   Build Options          --->                                ║
║   ───────────────────────────────────────────────────────    ║
║   Press S to save  |  Q to quit without saving               ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Use arrows to navigate, Space/Enter to toggle, S to save   ║
╠══════════════════════════════════════════════════════════════╣
║ <Select>    <Exit>    <Help>    <Save>    <Load>             ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Controls

| Key | Action |
|---|---|
| `↑` `↓` | Move between options |
| `Enter` or `Space` | Toggle checkbox / Enter submenu |
| `←` `→` | Decrease / Increase number or float values |
| `Y` | Enable a checkbox `[*]` |
| `N` | Disable a checkbox `[ ]` |
| `S` | Save configuration |
| `Q` or `Esc` | Go back / Quit without saving |
| `H` or `?` | Show help |

---

## Configuration Sections

### Performance Settings

Controls CPU and GPU acceleration.

| Option | Default | Description |
|---|---|---|
| Enable OpenMP multi-core | ON | Parallelize tensor operations across CPU cores |
| Thread Allocation | Auto | Auto = detect all cores, Manual = set count |
| Thread Count | 4 | Number of threads (Manual mode only) |
| Enable OpenCL GPU | OFF | Run tensor ops on GPU via OpenCL |
| Enable BLAS integration | OFF | Use OpenBLAS for ultra-fast matrix multiply |

**Thread Allocation — Auto vs Manual:**

```
Auto mode:
  neuralc detects your CPU core count at runtime
  Best for most users — works everywhere

Manual mode:
  You set exactly how many threads to use
  Useful when you want to reserve cores for other tasks
  Example: 4-core CPU, set to 3 to keep 1 core free
```

**To enable OpenCL GPU** you also need:
```bash
sudo apt install opencl-headers ocl-icd-opencl-dev
make gpu
```

---

### Training Defaults

Default hyperparameters used when no explicit values are passed.

| Option | Default | Description |
|---|---|---|
| Default Batch Size | 64 | Samples per gradient update |
| Default Learning Rate | 0.001 | Step size for gradient descent |
| Default Epochs | 20 | Full passes over training data |
| Default Optimizer | Adam | Adam / SGD / RMSProp |
| Default Dropout Rate | 0.3 | Fraction of neurons dropped in training |
| Enable Gradient Clipping | ON | Prevents exploding gradients |
| Gradient Clip Norm | 1.0 | Max allowed gradient L2 norm |

**Choosing an optimizer:**

```
Adam     — Best default. Adaptive learning rate. Works well out of the box.
SGD      — Simpler. Needs careful learning rate tuning. Good with momentum.
RMSProp  — Good for RNN/LSTM. Adapts to recent gradient magnitudes.
```

**Gradient Clipping** is especially important for:
- RNN / LSTM training (gradients can explode)
- Deep networks (many layers)
- Large learning rates

Typical values:
- RNN / LSTM: `max_norm = 1.0`
- Dense networks: `max_norm = 5.0`

---

### Memory Settings

Controls how neuralc allocates memory.

| Option | Default | Description |
|---|---|---|
| Memory Allocator | malloc | Standard C malloc / Pool allocator |
| Memory Pool Size | 512MB | Pre-allocated pool size |

**malloc** — Standard C allocation. Simple, works everywhere.

**Pool allocator** — Pre-allocates a large block at startup, then sub-allocates from it. Much faster for large numbers of small tensor allocations. Set pool size to at least 2x your largest model size.

---

### Debug & Profiling

| Option | Default | Description |
|---|---|---|
| Debug Mode | OFF | Verbose logging during training |
| Check for NaN | OFF | Detect NaN/Inf in tensor values |
| Profiling | OFF | Print timing info per operation |

**When to enable each:**

```
Debug Mode   — Enable when something isn't training correctly.
               Shows forward/backward pass details.
               Warning: significantly slows training.

NaN Check    — Enable when loss shows 'nan' or 'inf'.
               Catches numerical instability early.
               Small slowdown.

Profiling    — Enable to find which operations are slowest.
               Useful before optimizing performance.
               Small overhead.
```

---

### Build Options

Controls how the C compiler builds neuralc.

| Option | Default | Description |
|---|---|---|
| Optimization Level | -O2 | Compiler optimization flag |
| Enable AVX/SIMD | OFF | CPU vector instructions |
| Enable LTO | OFF | Link-time optimization |

**Optimization levels:**

```
-O0  No optimization. Fastest compile. Use for debugging.
-O1  Basic optimizations. Good balance for development.
-O2  Standard optimizations. Recommended for most users.
-O3  Aggressive optimizations. Fastest runtime, longer compile.
     Warning: may produce unexpected results on some code.
```

**AVX/SIMD** uses CPU vector instructions to process 4-8 floats simultaneously. Requires a modern CPU (2013+). Check if your CPU supports it:
```bash
grep -o 'avx[^ ]*' /proc/cpuinfo | head -5
```

**LTO (Link-Time Optimization)** allows the compiler to optimize across all `.c` files together. Slower to compile, faster to run.

---

## Generated File — `neuralc_config.h`

After saving, the UI generates `neuralc_config.h`:

```c
/*
 * neuralc_config.h — Auto-generated by neuralc config UI
 * Do not edit manually — run: make config
 */
#ifndef NEURALC_CONFIG_H
#define NEURALC_CONFIG_H

/* ── Performance ────────────────────────────────────── */
#define NEURALC_USE_OMP         1
#define NEURALC_OMP_AUTO        1
#define NEURALC_OMP_THREADS     4
#define NEURALC_USE_GPU         0
#define NEURALC_USE_BLAS        0

/* ── Training Defaults ──────────────────────────────── */
#define NEURALC_BATCH_SIZE      64
#define NEURALC_LR              0.001000f
#define NEURALC_OPTIMIZER       ADAM
#define NEURALC_DROPOUT         0.3000f
#define NEURALC_EPOCHS          20
#define NEURALC_GRAD_CLIP       1.0000f
#define NEURALC_USE_GRAD_CLIP   1

/* ── Memory ─────────────────────────────────────────── */
#define NEURALC_ALLOCATOR       0
#define NEURALC_POOL_MB         512

/* ── Debug ──────────────────────────────────────────── */
#define NEURALC_DEBUG           0
#define NEURALC_CHECK_NAN       0
#define NEURALC_PROFILE         0

/* ── Build ──────────────────────────────────────────── */
#define NEURALC_OPT_LEVEL       "-O2"
#define NEURALC_AVX             0
#define NEURALC_LTO             0

#endif /* NEURALC_CONFIG_H */
```

---

## Using Config in Your Code

```c
#include "neuralc_config.h"
#include "nn.h"
#include "optimizer.h"

// Use config values in your training
Adam *opt = adam_create(NEURALC_LR, 0.9f, 0.999f, 1e-8f, 0.0f);

for (int ep = 0; ep < NEURALC_EPOCHS; ep++) {
    float loss = nn_train_step(net, X, Y,
                               LOSS_CROSS_ENTROPY,
                               pred, grad);
    if (NEURALC_USE_GRAD_CLIP)
        nn_clip_gradients(net, NEURALC_GRAD_CLIP);
    adam_step(opt, net);
}
```

---

## Common Configurations

### Maximum Speed (CPU only)
```
Performance → OpenMP: ON, Thread Allocation: Auto
Build → Optimization: -O3, AVX: ON, LTO: ON
```

### GPU Training
```
Performance → GPU (OpenCL): ON
Build → Optimization: -O2
```

### Debugging a training problem
```
Debug → Debug Mode: ON, NaN Check: ON
Build → Optimization: -O0
Training → Gradient Clipping: ON, Norm: 1.0
```

### RNN / LSTM training
```
Training → Optimizer: Adam, LR: 0.001
Training → Gradient Clipping: ON, Norm: 1.0
Training → Dropout: 0.3
```

### Production / deployment
```
Performance → OpenMP: ON
Build → Optimization: -O3, LTO: ON
Debug → all OFF
```

---

## Files

| File | Purpose |
|---|---|
| `config_ui.c` | Terminal UI implementation (ANSI + termios) |
| `config_ui.h` | Config structs and API |
| `neuralc_config_main.c` | Entry point for `./neuralc_config` |
| `neuralc_config.h` | Generated output — include in your code |

---

## No External Dependencies

The config UI uses only:
- **ANSI escape codes** — colors and cursor positioning
- **POSIX termios** — raw keyboard input
- **Standard C** — no ncurses, no external libraries

It works on any Linux or macOS terminal out of the box.

---

## Contributor Ideas

Want to improve the config UI? Good first issues:

- Add a search feature (`/` to search options like vim)
- Add a diff view showing what changed from defaults
- Export config as a shell script for CI/CD
- Add preset profiles (speed / debug / RNN / GPU)
- Port to Windows (replace termios with Windows Console API)

---

*neuralc — Deep learning in pure C. Configure it your way.*
