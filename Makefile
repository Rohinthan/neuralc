CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

# ── auto-load neuralc_config.h if it exists ───────────────────────
ifneq (,$(wildcard neuralc_config.h))
  CFLAGS  += -DNEURALC_HAS_CONFIG
  NEURALC_USE_OMP := $(shell grep 'NEURALC_USE_OMP ' neuralc_config.h | awk '{print $$3}')
  NEURALC_USE_GPU := $(shell grep 'NEURALC_USE_GPU ' neuralc_config.h | awk '{print $$3}')
  NEURALC_OPT     := $(shell grep 'NEURALC_OPT_LEVEL' neuralc_config.h | awk '{print $$3}' | tr -d '"')
  ifeq ($(NEURALC_USE_OMP),1)
    CFLAGS  += -DUSE_OMP -fopenmp
    LDFLAGS += -fopenmp
  endif
  ifeq ($(NEURALC_USE_GPU),1)
    CFLAGS  += -DUSE_OPENCL -Igpu
    LDFLAGS += -lOpenCL
  endif
  ifneq ($(NEURALC_OPT),)
    CFLAGS  := $(filter-out -O2,$(CFLAGS)) $(NEURALC_OPT)
  endif
  $(info [neuralc] Using neuralc_config.h — OMP=$(NEURALC_USE_OMP) GPU=$(NEURALC_USE_GPU) OPT=$(NEURALC_OPT))
endif

SRC     = tensor.c layer.c nn.c optimizer.c dataloader.c \
          dropout.c batchnorm.c conv.c rnn.c mnist.c neuralc_init.c
OBJ     = $(SRC:.c=.o)

.PHONY: all demo rnn_demo mnist_demo demo_mnist config clean libneuralc omp gpu

all: neuralc demo rnn_demo mnist_demo

neuralc: $(OBJ) main.o
	$(CC) $(CFLAGS) -o neuralc $(OBJ) main.o $(LDFLAGS)

demo: $(OBJ) demo.o
	$(CC) $(CFLAGS) -o demo $(OBJ) demo.o $(LDFLAGS)

rnn_demo: $(OBJ) demo_rnn.o
	$(CC) $(CFLAGS) -o rnn_demo $(OBJ) demo_rnn.o $(LDFLAGS)

mnist_demo demo_mnist: $(OBJ) demo_mnist.o
	$(CC) $(CFLAGS) -o mnist_demo $(OBJ) demo_mnist.o $(LDFLAGS)

# ── neuralc config UI ─────────────────────────────────────────────
config: config_ui.o neuralc_config_main.o
	$(CC) $(CFLAGS) -o neuralc_config config_ui.o neuralc_config_main.o $(LDFLAGS)
	@echo ""
	@echo "✓ Config UI built! Run: ./neuralc_config"
	@echo "  Then: make clean && make"
	@echo ""

config_ui.o: config_ui.c config_ui.h
	$(CC) $(CFLAGS) -c config_ui.c -o config_ui.o

neuralc_config_main.o: neuralc_config_main.c config_ui.h
	$(CC) $(CFLAGS) -c neuralc_config_main.c -o neuralc_config_main.o

libneuralc: $(SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o libneuralc.so $(SRC) $(LDFLAGS)
	@echo "Built libneuralc.so"

omp: CFLAGS  += -DUSE_OMP -fopenmp
omp: LDFLAGS += -fopenmp
omp: all
	@echo "Built with OpenMP"

gpu: CFLAGS  += -DUSE_OPENCL -Igpu
gpu: LDFLAGS += -lOpenCL
gpu: $(OBJ) main.o gpu/neuralc_gpu.o
	$(CC) $(CFLAGS) -o neuralc_gpu $(OBJ) main.o gpu/neuralc_gpu.o $(LDFLAGS)

gpu/neuralc_gpu.o: gpu/neuralc_gpu.c gpu/neuralc_gpu.h
	$(CC) $(CFLAGS) -DUSE_OPENCL -c gpu/neuralc_gpu.c -o gpu/neuralc_gpu.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) main.o demo.o demo_rnn.o demo_mnist.o \
	      config_ui.o neuralc_config_main.o neuralc_init.o \
	      neuralc demo rnn_demo mnist_demo libneuralc.so \
	      neuralc_config xor_weights.bin mnist_best.bin \
	      gpu/neuralc_gpu.o neuralc_gpu test_rnn_min
