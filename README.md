# NeuralC

A lightweight neural network library written entirely in C.

## Overview

NeuralC is an experimental deep learning library inspired by frameworks like PyTorch and TensorFlow — but built directly in C for performance, simplicity, and low-level control.

While Python dominates AI development, most high-performance libraries are implemented in C/C++ under the hood. NeuralC explores the idea of building a neural network framework directly in C to better understand and leverage hardware-level efficiency.

## Motivation

* Learn how deep learning frameworks work internally
* Build a minimal and efficient neural network system
* Explore performance benefits of low-level programming
* Create a foundation for future optimization (SIMD, GPU, etc.)

## Features

* Fully connected (Dense) layers
* Activation functions (ReLU, Sigmoid, etc.)
* Loss functions (MSE, BCE, Cross Entropy)
* Optimizers:

  * SGD (with momentum)
  * Adam (with bias correction)
* Learning rate schedulers
* Simple training loop support
* Model save/load functionality

## Example

```c
Network *net = network_create();
network_add_dense(net, 2, 4, ACT_RELU);
network_add_dense(net, 4, 1, ACT_SIGMOID);

Adam *opt = adam_create(0.05f, 0.9f, 0.999f, 1e-8f, 1e-4f);

for (int epoch = 0; epoch < 5000; epoch++) {
    network_forward(net, X);
    float loss = bce_loss(net->output, Y);
    network_backward(net, Y);
    adam_step(opt, net);
}
```

## Project Structure

```
.
├── nn.c / nn.h          # Network management
├── layer.c / layer.h    # Layer implementations
├── optimizer.c / .h     # Optimizers (SGD, Adam)
├── loss.c / loss.h      # Loss functions
├── demo.c               # Example training (XOR)
```

## Note

This project started as an experimental idea and the initial version was developed with AI-assisted coding tools. The goal is to refine, optimize, and expand it into a fully functional neural network framework.

## Roadmap

* [ ] Improve tensor operations (broadcasting, reshape)
* [ ] Add convolutional layers
* [ ] Implement automatic differentiation (autograd)
* [ ] Performance optimization (SIMD / BLAS)
* [ ] GPU support (CUDA/OpenCL)
* [ ] Build system (CMake)

## Contributing

Contributions are welcome!
Feel free to open issues, suggest features, or submit pull requests.

## Support

If you like this project, consider giving it a star!
