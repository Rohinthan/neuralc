CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
CFLAGS += -I include/     # finds all headers in include/
CFLAGS += -I config/      # finds config_ui.h and neuralc_init.h
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

# ── sources ───────────────────────────────────────────────────────
SRC = src/tensor.c       \
      src/layer.c        \
      src/nn.c           \
      src/optimizer.c    \
      src/dataloader.c   \
      src/dropout.c      \
      src/batchnorm.c    \
      src/conv.c         \
      src/rnn.c          \
      src/mnist.c        \
      config/neuralc_init.c

OBJ = $(patsubst src/%.c,   build/%.o, $(filter src/%,   $(SRC))) \
      $(patsubst config/%.c, build/%.o, $(filter config/%, $(SRC)))

.PHONY: all demo rnn_demo mnist_demo demo_mnist config \
        clean libneuralc omp gpu test

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

# ── test ──────────────────────────────────────────────────────────
test: $(OBJ) build/test_tensor.o
	$(CC) $(CFLAGS) -o test_tensor $(OBJ) build/test_tensor.o $(LDFLAGS)
	@echo "Running tests..."
	@./test_tensor

# ── menuconfig ────────────────────────────────────────────────────
config: menuconfig
	@./menuconfig

menuconfig: build/config_ui.o build/neuralc_config_main.o
	$(CC) $(CFLAGS) -o menuconfig \
	      build/config_ui.o build/neuralc_config_main.o $(LDFLAGS)

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

# ── shared library ────────────────────────────────────────────────
libneuralc: $(SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o libneuralc.so $(SRC) $(LDFLAGS)
	@echo "Built libneuralc.so"

# ── omp / gpu ─────────────────────────────────────────────────────
omp: CFLAGS  += -DUSE_OMP -fopenmp
omp: LDFLAGS += -fopenmp
omp: all

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
	@echo "✓ Clean done"
