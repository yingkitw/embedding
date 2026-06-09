.PHONY: all build test clean install example help

# Default target
all: build

# Build the project
build:
	@echo "Building embedding-trainer..."
	cargo build --release
	@echo "Build completed!"

# Run tests
test:
	@echo "Running tests..."
	cargo test
	@echo "Tests completed!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf build/
	@echo "Clean completed!"

# Install CLI tool
install:
	@echo "Installing embedding-trainer..."
	cargo install --path .
	@echo "Installation completed!"

# Create example model and test
example:
	@echo "Creating example model..."
	mkdir -p build
	./target/release/embedding-train train \
		--input example_data.txt \
		--output build/example_model.json \
		--embeddings build/example_embeddings.txt \
		--dim 100 \
		--epochs 5 \
		--model-type skipgram
	@echo "Example model created!"

	@echo "Testing similarity query..."
	./target/release/embedding-train similarity "fox" "dog" \
		--model build/example_model.json \
		--vocab build/example_model.json
	@echo "Example completed!"

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build the project (default)"
	@echo "  build      - Build the project"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean build artifacts"
	@echo "  install    - Install CLI tool"
	@echo "  example    - Create example model and test"
	@echo "  help       - Show this help message"

# Run the test script
test-cli:
	@echo "Running CLI test script..."
	rustc test_cli.rs -o test_cli
	./test_cli
	rm test_cli