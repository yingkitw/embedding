#!/bin/bash

# Build script for embedding-trainer

set -e

echo "Building embedding-trainer..."

# Create build directory
mkdir -p build

# Build the project
cargo build --release

echo "Build completed successfully!"

# Create symlink for easier CLI usage
ln -sf "$(pwd)/target/release/embedding-train" "$(pwd)/build/embedding-train"

echo "CLI tool available at: build/embedding-train"

# Run tests if requested
if [ "$1" = "--test" ]; then
    echo "Running tests..."
    cargo test
    echo "Tests completed!"
fi

# Create example model if requested
if [ "$1" = "--example" ]; then
    echo "Creating example model..."
    ./build/embedding-train train \
        --input example_data.txt \
        --output build/example_model.json \
        --embeddings build/example_embeddings.txt \
        --dim 100 \
        --epochs 5 \
        --model-type skipgram
    
    echo "Example model created at: build/example_model.json"
    echo "Example embeddings created at: build/example_embeddings.txt"
    
    # Test similarity query
    echo "Testing similarity query..."
    ./build/embedding-train similarity "fox" "dog" \
        --model build/example_model.json \
        --vocab build/example_model.json
    
    echo "Example completed!"
fi