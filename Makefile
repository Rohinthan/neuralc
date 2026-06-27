CC      = gcc
# Added -Iinclude so gcc searches the include/ folder automatically for headers!
CFLAGS  = -O2 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

# ── auto-load neuralc_config.h if it exists ───────────────────────
ifneq (,$(wildcard neuralc_config.h))
  CFLAGS  += -DNEURALC_HAS_CONFIG
  NEURALC_USE_OMP := $(shell grep 'NEURALC_USE_OMP ' neuralc_config.h | awk '{print $$3}')
  NEURALC_USE_GPU := $(shell grep 'NEURALC_USE_GPU ' neuralc_config.h | awk '{print $$3}')
  NEURALC_OPT     := $(shell grep 'NEURALC_OPT_LEVEL' neuralc_config.h | awk '{print $$3}' | tr -d '\"')
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
  $(info [neuralc] Config loaded — OMP=$(NEURALC_USE_OMP) GPU=$(NEURALC_USE_GPU) OPT=$(NEURALC_OPT))
endif

# Updated paths to target files living inside the src/ directory
SRC     = src/tensor.c src/layer.c src/nn.c src/optimizer.c src/dataloader.c \
          src/dropout.c src/batchnorm.c src/conv.c src/neuralc_init.c

# Automatically swap out src/%.c for build/%.o to keep project root clean
OBJ     = $(SRC:src/%.c=build/%.o)

all: directories mnist_demo

# Ensure a dedicated build folder always exists for your compiled objects
directories:
	@mkdir -p build

# Generic compilation rule to map everything inside src/ directly into build/
build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── Executable targets ───────────────────────────────────────────
mnist_demo: $(OBJ) build/demo_mnist.o
	$(CC) $(CFLAGS) -o mnist_demo $(OBJ) build/demo_mnist.o $(LDFLAGS)

build/demo_mnist.o: src/demo_mnist.c
	$(CC) $(CFLAGS) -c src/demo_mnist.c -o build/demo_mnist.o

# ── menuconfig UI targets ────────────────────────────────────────
config: menuconfig
	@./menuconfig

menuconfig: build/config_ui.o build/neuralc_config_main.o
	$(CC) $(CFLAGS) -o menuconfig build/config_ui.o build/neuralc_config_main.o $(LDFLAGS)

build/config_ui.o: src/config_ui.c
	$(CC) $(CFLAGS) -c src/config_ui.c -o build/config_ui.o

build/neuralc_config_main.o: src/neuralc_config_main.c
	$(CC) $(CFLAGS) -c src/neuralc_config_main.c -o build/neuralc_config_main.o

clean:
	rm -rf build menuconfig mnist_demo neuralc_config.h
