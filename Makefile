CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

SRC     = tensor.c layer.c nn.c optimizer.c dataloader.c \
          dropout.c batchnorm.c conv.c rnn.c mnist.c
OBJ     = $(SRC:.c=.o)

.PHONY: all demo rnn_demo mnist_demo clean libneuralc omp gpu

all: neuralc demo rnn_demo mnist_demo

neuralc: $(OBJ) main.o
	$(CC) $(CFLAGS) -o neuralc $(OBJ) main.o $(LDFLAGS)

demo: $(OBJ) demo.o
	$(CC) $(CFLAGS) -o demo $(OBJ) demo.o $(LDFLAGS)

rnn_demo: $(OBJ) demo_rnn.o
	$(CC) $(CFLAGS) -o rnn_demo $(OBJ) demo_rnn.o $(LDFLAGS)

mnist_demo: $(OBJ) demo_mnist.o
	$(CC) $(CFLAGS) -o mnist_demo $(OBJ) demo_mnist.o $(LDFLAGS)

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
	      neuralc demo rnn_demo mnist_demo libneuralc.so \
	      xor_weights.bin mnist_best.bin gpu/neuralc_gpu.o \
	      neuralc_gpu test_rnn_min
