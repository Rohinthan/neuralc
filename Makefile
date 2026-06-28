CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
CFLAGS += -I include/          # tell compiler where .h files are
LDFLAGS = -lm

# ── auto-load neuralc_config.h if it exists ───────────────────────
ifneq (,$(wildcard neuralc_config.h))
  CFLAGS += -DNEURALC_HAS_CONFIG
  NEURALC_USE_OMP := $(shell grep 'NEURALC_USE_OMP ' neuralc_config.h | awk '{print $$3}')
  NEURALC_USE_GPU := $(shell grep 'NEURALC_USE_GPU ' neuralc_config.h | awk '{print $$3}')
  NEURALC_OPT     := $(shell grep 'NEURALC_OPT_LEVEL' neuralc_config.h | awk '{print $$3}' | tr -d '"')
  ifeq ($(NEURALC_USE_OMP),1)
    CFLAGS  += -DUSE_OMP -fopenmp
    LDFLAGS += -fopenmp
  endif
  ifeq ($(NEURALC_USE_GPU),1)
    CFLAGS  += -DUSE_OPENCL -I include/gpu
    LDFLAGS += -lOpenCL
  endif
  ifneq ($(NEURALC_OPT),)
    CFLAGS := $(filter-out -O2,$(CFLAGS)) $(NEURALC_OPT)
  endif
  $(info [neuralc] Config loaded — OMP=$(NEURALC_USE_OMP) GPU=$(NEURALC_USE_GPU) OPT=$(NEURALC_OPT))
endif

# ── source files (now in src/) ────────────────────────────────────
SRC = src/tensor.c      \
      src/layer.c       \
      src/nn.c          \
      src/optimizer.c   \
      src/dataloader.c  \
      src/dropout.c     \
      src/batchnorm.c   \
      src/conv.c        \
      src/rnn.c         \
      src/mnist.c       \
      src/neuralc_init.c

# ── object files go into build/ ───────────────────────────────────
OBJ = $(SRC:src/%.c=build/%.o)

.PHONY: all demo rnn_demo mnist_demo demo_mnist config clean \
        libneuralc omp gpu test

# ── create build/ folder if it doesn't exist ──────────────────────
$(shell mkdir -p build)

# ── main targets ──────────────────────────────────────────────────
all: neuralc demo rnn_demo mnist_demo

neuralc: $(OBJ) build/main.o
	$(CC) $(CFLAGS) -o neuralc $(OBJ) build/main.o $(LDFLAGS)

demo: $(OBJ) build/demo.o
	$(CC) $(CFLAGS) -o demo $(OBJ) build/demo.o $(LDFLAGS)

rnn_demo: $(OBJ) build/demo_rnn.o
	$(CC) $(CFLAGS) -o rnn_demo $(OBJ) build/demo_rnn.o $(LDFLAGS)

mnist_demo demo_mnist: $(OBJ) build/demo_mnist.o
	$(CC) $(CFLAGS) -o mnist_demo $(OBJ) build/demo_mnist.o $(LDFLAGS)

# ── tests ─────────────────────────────────────────────────────────
test: $(OBJ) build/test_tensor.o
	$(CC) $(CFLAGS) -o test_tensor $(OBJ) build/test_tensor.o $(LDFLAGS)
	@echo "Running tests..."
	@./test_tensor

# ── compile src/*.c → build/*.o ───────────────────────────────────
build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── compile examples/*.c → build/*.o ──────────────────────────────
build/%.o: examples/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── compile tests/*.c → build/*.o ─────────────────────────────────
build/%.o: tests/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── old-style root .c files (backwards compat) ────────────────────
build/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── menuconfig ────────────────────────────────────────────────────
config: menuconfig
	@./menuconfig

menuconfig: build/config_ui.o build/neuralc_config_main.o
	$(CC) $(CFLAGS) -o menuconfig \
	      build/config_ui.o build/neuralc_config_main.o $(LDFLAGS)

build/config_ui.o: src/config_ui.c include/config_ui.h
	$(CC) $(CFLAGS) -c src/config_ui.c -o build/config_ui.o

build/neuralc_config_main.o: src/neuralc_config_main.c include/config_ui.h
	$(CC) $(CFLAGS) -c src/neuralc_config_main.c \
	      -o build/neuralc_config_main.o

# ── shared library for Python bindings ───────────────────────────
libneuralc: $(SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o libneuralc.so $(SRC) $(LDFLAGS)
	@echo "Built libneuralc.so — run: python3 python/neuralc.py"

# ── OpenMP build ──────────────────────────────────────────────────
omp: CFLAGS  += -DUSE_OMP -fopenmp
omp: LDFLAGS += -fopenmp
omp: all
	@echo "Built with OpenMP multi-core"

# ── OpenCL GPU build ──────────────────────────────────────────────
gpu: CFLAGS  += -DUSE_OPENCL -I include/gpu
gpu: LDFLAGS += -lOpenCL
gpu: $(OBJ) build/main.o build/gpu_backend.o
	$(CC) $(CFLAGS) -o neuralc_gpu \
	      $(OBJ) build/main.o build/gpu_backend.o $(LDFLAGS)

build/gpu_backend.o: src/gpu/neuralc_gpu.c include/gpu/neuralc_gpu.h
	$(CC) $(CFLAGS) -DUSE_OPENCL -c src/gpu/neuralc_gpu.c \
	      -o build/gpu_backend.o

# ── clean ─────────────────────────────────────────────────────────
clean:
	rm -f build/*.o
	rm -f neuralc demo rnn_demo mnist_demo test_tensor
	rm -f menuconfig libneuralc.so neuralc_gpu
	rm -f xor_weights.bin mnist_best.bin
	@echo "Clean done"
