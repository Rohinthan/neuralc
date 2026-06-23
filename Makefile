CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

SRC     = tensor.c layer.c nn.c optimizer.c dataloader.c \
          dropout.c batchnorm.c conv.c
OBJ     = $(SRC:.c=.o)

.PHONY: all demo clean libneuralc gpu

# ── default: build everything ──────────────────────────────────────
all: neuralc demo

# ── main demo binary ───────────────────────────────────────────────
neuralc: $(OBJ) main.o
	$(CC) $(CFLAGS) -o neuralc $(OBJ) main.o $(LDFLAGS)

# ── xor demo ──────────────────────────────────────────────────────
demo: $(OBJ) demo.o
	$(CC) $(CFLAGS) -o demo $(OBJ) demo.o $(LDFLAGS)

# ── shared library for Python ctypes bindings ─────────────────────
libneuralc: $(SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o libneuralc.so $(SRC) $(LDFLAGS)
	@echo "Built libneuralc.so"
	@echo "Run: python3 python/neuralc.py"

# ── OpenMP build (multi-core) ─────────────────────────────────────
omp: CFLAGS += -DUSE_OMP -fopenmp
omp: LDFLAGS += -fopenmp
omp: all
	@echo "Built with OpenMP multi-core support"

# ── OpenCL GPU build ──────────────────────────────────────────────
gpu: CFLAGS  += -DUSE_OPENCL -Igpu
gpu: LDFLAGS += -lOpenCL
gpu: GPU_OBJ  = gpu/neuralc_gpu.o
gpu: $(OBJ) main.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -o neuralc_gpu $(OBJ) main.o $(GPU_OBJ) $(LDFLAGS)
	@echo "Built neuralc_gpu with OpenCL support"

gpu/neuralc_gpu.o: gpu/neuralc_gpu.c gpu/neuralc_gpu.h
	$(CC) $(CFLAGS) -DUSE_OPENCL -c gpu/neuralc_gpu.c -o gpu/neuralc_gpu.o

# ── OpenMP + GPU combined ─────────────────────────────────────────
full: CFLAGS  += -DUSE_OMP -fopenmp -DUSE_OPENCL -Igpu
full: LDFLAGS += -fopenmp -lOpenCL
full: $(OBJ) main.o gpu/neuralc_gpu.o
	$(CC) $(CFLAGS) -o neuralc_full $(OBJ) main.o \
	      gpu/neuralc_gpu.o $(LDFLAGS)
	@echo "Built with OpenMP + OpenCL"

# ── compile rules ─────────────────────────────────────────────────
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── clean ─────────────────────────────────────────────────────────
clean:
	rm -f $(OBJ) main.o demo.o neuralc demo \
	      libneuralc.so xor_weights.bin \
	      gpu/neuralc_gpu.o neuralc_gpu neuralc_full
