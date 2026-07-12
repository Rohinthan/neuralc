CC      = gcc
NVCC    ?= nvcc
CUDA_PATH ?= /usr/local/cuda

CFLAGS  = -O2 -Wall -Wextra -std=c11 -fPIC
CFLAGS += -I include/
CFLAGS += -I config/
CFLAGS += -I .
LDFLAGS = -lm

# GPU_LDFLAGS is deliberately kept SEPARATE from LDFLAGS — it must
# only be added to the binaries that actually link GPU_OBJS (see
# below). If it were folded into the global LDFLAGS, `make config`
# (which links the standalone `menuconfig` tool) would fail on any
# machine without libcudart/libOpenCL installed — exactly the
# machine you need menuconfig on to turn GPU support OFF in the
# first place.
GPU_LDFLAGS =
GPU_OBJS    =

# ── auto-load neuralc_config.h if it exists ───────────────────────
ifneq (,$(wildcard neuralc_config.h))
  CFLAGS += -DNEURALC_HAS_CONFIG
  NEURALC_USE_OMP     := $(shell grep 'NEURALC_USE_OMP '     neuralc_config.h | awk '{print $$3}')
  NEURALC_GPU_BACKEND := $(shell grep 'NEURALC_GPU_BACKEND ' neuralc_config.h | awk '{print $$3}')
  NEURALC_OPT         := $(shell grep 'NEURALC_OPT_LEVEL'    neuralc_config.h | awk '{print $$3}' | tr -d '"')
  ifeq ($(NEURALC_USE_OMP),1)
    CFLAGS  += -DUSE_OMP -fopenmp
    LDFLAGS += -fopenmp
  endif
  # NEURALC_GPU_BACKEND: 0=Off 1=OpenCL 2=CUDA 3=Both (set via
  # `make config` — see config/config_ui.c's GPU_BACKEND radio item)
  ifeq ($(NEURALC_GPU_BACKEND),1)
    CFLAGS      += -DUSE_OPENCL -I include/gpu
    GPU_LDFLAGS += -lOpenCL
    GPU_OBJS    += build/opencl_backend.o
  endif
  ifeq ($(NEURALC_GPU_BACKEND),2)
    CFLAGS      += -DUSE_CUDA -I include/gpu
    GPU_LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lstdc++
    GPU_OBJS    += build/cuda_backend.o
  endif
  ifeq ($(NEURALC_GPU_BACKEND),3)
    CFLAGS      += -DUSE_CUDA -DUSE_OPENCL -I include/gpu
    GPU_LDFLAGS += -lOpenCL -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lstdc++
    GPU_OBJS    += build/opencl_backend.o build/cuda_backend.o
  endif
  ifneq ($(NEURALC_OPT),)
    CFLAGS := $(filter-out -O2,$(CFLAGS)) $(NEURALC_OPT)
  endif
  $(info [neuralc] Config loaded - OMP=$(NEURALC_USE_OMP) GPU_BACKEND=$(NEURALC_GPU_BACKEND) OPT=$(NEURALC_OPT))
endif

# ── sources ───────────────────────────────────────────────────────
SRC_CORE = src/tensor.c    \
           src/layer.c     \
           src/nn.c        \
           src/optimizer.c \
           src/dataloader.c\
           src/dropout.c   \
           src/batchnorm.c \
           src/conv.c      \
           src/rnn.c       \
           src/mnist.c

SRC_CFG  = config/neuralc_init.c
SRC      = $(SRC_CORE) $(SRC_CFG)

OBJ_CORE = $(patsubst src/%.c,    build/%.o, $(SRC_CORE))
OBJ_CFG  = $(patsubst config/%.c, build/%.o, $(SRC_CFG))
# GPU_OBJS (cuda_backend.o / opencl_backend.o) is populated above,
# per whatever NEURALC_GPU_BACKEND says — empty when GPU is Off.
OBJ      = $(OBJ_CORE) $(OBJ_CFG) $(GPU_OBJS)

.PHONY: all demo rnn_demo mnist_demo demo_mnist cnn_mnist config \
        clean libneuralc omp test

$(shell mkdir -p build)

# ── main targets ──────────────────────────────────────────────────
# NOTE: $(GPU_LDFLAGS) is appended explicitly here (and nowhere near
# the menuconfig rule below) — see the comment on GPU_LDFLAGS above.
all: neuralc demo rnn_demo mnist_demo

neuralc: $(OBJ) build/main.o
	$(CC) $(CFLAGS) -o neuralc $(OBJ) build/main.o $(LDFLAGS) $(GPU_LDFLAGS)

demo: $(OBJ) build/demo.o
	$(CC) $(CFLAGS) -o demo $(OBJ) build/demo.o $(LDFLAGS) $(GPU_LDFLAGS)

rnn_demo: $(OBJ) build/demo_rnn.o
	$(CC) $(CFLAGS) -o rnn_demo $(OBJ) build/demo_rnn.o $(LDFLAGS) $(GPU_LDFLAGS)

mnist_demo demo_mnist: $(OBJ) build/demo_mnist.o
	$(CC) $(CFLAGS) -o mnist_demo $(OBJ) build/demo_mnist.o $(LDFLAGS) $(GPU_LDFLAGS)

cnn_mnist: $(OBJ) build/demo_cnn_mnist.o
	$(CC) $(CFLAGS) -o cnn_mnist $(OBJ) build/demo_cnn_mnist.o $(LDFLAGS) $(GPU_LDFLAGS)

# ── explicit compile rules for all example files ──────────────────
build/main.o: examples/main.c
	$(CC) $(CFLAGS) -c examples/main.c -o build/main.o

build/demo.o: examples/demo.c
	$(CC) $(CFLAGS) -c examples/demo.c -o build/demo.o

build/demo_rnn.o: examples/demo_rnn.c
	$(CC) $(CFLAGS) -c examples/demo_rnn.c -o build/demo_rnn.o

build/demo_mnist.o: examples/demo_mnist.c
	$(CC) $(CFLAGS) -c examples/demo_mnist.c -o build/demo_mnist.o

build/demo_cnn_mnist.o: examples/demo_cnn_mnist.c
	$(CC) $(CFLAGS) -c examples/demo_cnn_mnist.c -o build/demo_cnn_mnist.o

# ── tests ─────────────────────────────────────────────────────────
test: $(OBJ) build/test_tensor.o
	$(CC) $(CFLAGS) -o test_tensor $(OBJ) build/test_tensor.o $(LDFLAGS) $(GPU_LDFLAGS)
	@echo "Running tests..."
	@./test_tensor

# ── menuconfig ────────────────────────────────────────────────────
# Deliberately does NOT use $(GPU_LDFLAGS) or link $(GPU_OBJS) — this
# tool must build on a machine with no CUDA/OpenCL toolchain at all,
# since picking "Off" is one of the things it's for. config_ui.c only
# probes for hardware via file-existence checks, it never calls into
# cuda_backend.h/opencl_backend.h.
config: menuconfig
	@./menuconfig

menuconfig: build/config_ui.o build/neuralc_config_main.o
	$(CC) $(CFLAGS) -o menuconfig \
	      build/config_ui.o build/neuralc_config_main.o -lm

build/config_ui.o: config/config_ui.c config/config_ui.h
	$(CC) $(CFLAGS) -c config/config_ui.c -o build/config_ui.o

build/neuralc_config_main.o: config/neuralc_config_main.c config/config_ui.h
	$(CC) $(CFLAGS) -c config/neuralc_config_main.c \
	      -o build/neuralc_config_main.o

# ── compile rules ─────────────────────────────────────────────────
build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: config/%.c
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: examples/%.c
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: tests/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── GPU backend objects ─────────────────────────────────────────────
# Only built when NEURALC_GPU_BACKEND selects them (see auto-config
# block above) — these explicit rules take precedence over the
# generic src/%.c / src/%.cu pattern since there is none for .cu.
build/cuda_backend.o: src/gpu/cuda_backend.cu include/gpu/cuda_backend.h
	$(NVCC) -O3 -Xcompiler -fPIC -I include/gpu -c src/gpu/cuda_backend.cu -o build/cuda_backend.o

build/opencl_backend.o: src/gpu/opencl_backend.c include/gpu/opencl_backend.h
	$(CC) $(CFLAGS) -c src/gpu/opencl_backend.c -o build/opencl_backend.o

# ── shared library ────────────────────────────────────────────────
libneuralc: $(OBJ)
	$(CC) $(CFLAGS) -fPIC -shared -o libneuralc.so $(OBJ) $(LDFLAGS) $(GPU_LDFLAGS)
	@echo "Built libneuralc.so"

# ── omp ───────────────────────────────────────────────────────────
# Force OpenMP on without a config file. There's no equivalent `gpu:`
# convenience target any more — GPU backend selection now always
# goes through `make config` (NEURALC_GPU_BACKEND), since a plain
# `make gpu` can't express "CUDA vs OpenCL vs Both" as a single flag.
omp: CFLAGS  += -DUSE_OMP -fopenmp
omp: LDFLAGS += -fopenmp
omp: all

# ── clean ─────────────────────────────────────────────────────────
clean:
	rm -f build/*.o
	rm -f neuralc demo rnn_demo mnist_demo cnn_mnist test_tensor
	rm -f menuconfig libneuralc.so
	rm -f xor_weights.bin mnist_best.bin cnn_mnist_best.bin
	@echo "Clean done"
