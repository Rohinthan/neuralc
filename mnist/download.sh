#!/bin/bash
# download.sh — Download and extract MNIST dataset
# Run from the neuralc root directory: bash mnist/download.sh

set -e

mkdir -p mnist
cd mnist

BASE="http://yann.lecun.com/exdb/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

echo "Downloading MNIST dataset..."

for FILE in "${FILES[@]}"; do
    if [ ! -f "${FILE%.gz}" ]; then
        echo "  Downloading $FILE..."
        wget -q --show-progress "$BASE/$FILE" -O "$FILE"
        echo "  Extracting $FILE..."
        gunzip "$FILE"
    else
        echo "  Already exists: ${FILE%.gz}"
    fi
done

echo ""
echo "MNIST files ready:"
ls -lh *.ubyte
echo ""
echo "Now run: make mnist_demo && ./mnist_demo"
