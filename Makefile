CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

SRC     = tensor.c layer.c nn.c optimizer.c dataloader.c \
          dropout.c batchnorm.c conv.c
OBJ     = $(SRC:.c=.o)

.PHONY: all main demo clean

all: neuralc demo

neuralc: $(OBJ) main.o
	$(CC) $(CFLAGS) -o neuralc $(OBJ) main.o $(LDFLAGS)

demo: $(OBJ) demo.o
	$(CC) $(CFLAGS) -o demo $(OBJ) demo.o $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) main.o demo.o neuralc demo xor_weights.bin
