# Embedding Trainer

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.1.3-blue.svg)](https://crates.io/crates/embedding)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/yingkitw/embedding)

A fast and flexible Rust library and CLI tool for training word embeddings from scratch using Skip-gram and CBOW algorithms with built-in validation, evaluation, and semantic search.

## ✨ Features

### 🚀 **Algorithms**
- **Skip-gram**: Predicts context words given target words
- **CBOW**: Predicts target words given context words

### 📊 **Training Features**
- Configurable embedding dimensions
- Adjustable learning rates and epochs
- Customizable context windows
- Negative sampling support (uniform or unigram^0.75 distribution)
- Sub-sampling of frequent words (Mikolov-style)
- Learning rate warm-up for stabler early training
- Model checkpointing (save/resume every N epochs)
- Multi-threaded parallel training over CPU cores
- Batch processing capabilities
- Learning rate scheduling (constant, exponential, step, cosine)
- Early stopping with configurable patience
- L2 regularization and gradient clipping
- Train/validation split with metrics export
- Per-epoch training history / learning curves (JSON export)
- K-fold cross-validation with averaged metrics

### 🔧 **CLI Tools**
- **Training**: Train embeddings from text data with optional validation split
- **Similarity**: Calculate semantic similarity between words
- **Inspection**: Analyze trained models and vocabulary
- **Export**: Save embeddings in multiple formats (text, JSON, binary, Word2Vec, INT8/FP16 ONNX)
- **Validate**: Evaluate a saved model on held-out validation text
- **Interactive**: Query trained models interactively (similarity, analogy, search)

### � **Evaluation & Analysis**
- **Benchmarks**: Evaluate against built-in word similarity benchmarks (WordSim-353, SimLex-999, MEN, RW, SCWS) with Spearman correlation
- **Clustering**: K-means and hierarchical clustering of embeddings
- **Cross-validation**: K-fold cross-validation with per-fold metrics
- **Learning curves**: Per-epoch loss and learning rate tracking with JSON export

### � **Data Support**
- Text file processing with Unicode normalization
- Source code preprocessing (Rust, Python, JavaScript, etc.)
- BPE subword tokenization and FastText-style character n-grams
- WordPiece subword tokenization (BERT-style)
- Vocabulary management
- Model persistence
- Multiple export formats (JSON, binary, Word2Vec, ONNX, NumPy)
- Streaming support for large datasets
- Pluggable compute backend trait (CPU implemented, GPU ready)

### 🤖 **Advanced Models**
- **Transformer encoder**: Multi-head self-attention with position encoding for contextualized embeddings
- **Multi-modal fusion**: Concatenation, weighted average, attention fusion, projection fusion, cross-modal similarity
- **Real-time training**: Incremental updates and streaming micro-batch training without full retrain

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yingkitw/embedding.git
cd embedding

# Build the project
cargo build --release

# Or install locally
cargo install --path .
```

### GPU Acceleration (Optional)

Enable GPU compute via the `gpu` feature flag. This uses [wgpu](https://github.com/gfx-rs/wgpu) compute shaders and works on Vulkan, Metal, and DX12 backends without vendor-specific SDKs.

```bash
# Build with GPU support
cargo build --release --features gpu

# Install with GPU support
cargo install --path . --features gpu
```

When the `gpu` feature is enabled, `EmbeddingModel::new()` automatically selects the best available backend (GPU if present, otherwise CPU). You can also explicitly create a GPU backend:

```rust
use embedding::backend::{GpuBackend, Backend};

// Attempt GPU initialization; fails gracefully if no GPU is available
if let Ok(gpu) = GpuBackend::new() {
    println!("Using {} backend", gpu.name());
    let embeddings = gpu.init_embeddings(1000, 300);
}
```

> **Note:** GPU operations have CPU-GPU transfer overhead. For small models the CPU backend may still be faster. GPU acceleration shines with large batch matrix multiplications (`matmul`).

### Basic Usage

#### 1. Train Your First Embeddings

```bash
# Prepare your training data
echo "the quick brown fox jumps over the lazy dog" > data.txt

# Train embeddings using Skip-gram
embedding train \
    --input data.txt \
    --output model.json \
    --embeddings embeddings.txt \
    --dim 100 \
    --epochs 10 \
    --model-type skipgram

# Train with validation split
embedding train \
    --input data.txt \
    --output model.json \
    --embeddings embeddings.txt \
    --dim 100 \
    --epochs 10 \
    --validation-ratio 0.2 \
    --validation-output metrics.json
```

#### 2. Calculate Similarity

```bash
# Calculate similarity between words
embedding similarity fox dog \
    --model model.json

# Expected output:
# Similarity between 'fox' and 'dog': 0.8234
```

#### 3. Inspect Model

```bash
# View model information
embedding info --model model.json

# Shows vocabulary size, embedding dimension, training config
```

#### 4. Export Embeddings

```bash
# Export to different formats
embedding export \
    --model model.json \
    --output embeddings.json \
    --format json
```

## 📚 Library Usage

### Basic Example

```rust
use embedding::*;

fn main() -> Result<(), String> {
    // Load and prepare data in one step
    let data = TrainingData::from_text("the quick brown fox jumps over the lazy dog");

    // Configure training with sensible defaults and fluent setters
    let config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(300)
        .with_epochs(10);

    // Train model
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data)?;

    // Calculate similarity
    if let Some(similarity) = model.similarity("fox", "dog", &data) {
        println!("Similarity: {:.4}", similarity);
    }

    // Save model
    model.save_embeddings("embeddings.txt", &data)?;

    Ok(())
}
```

### Advanced Usage

```rust
use embedding::*;

fn advanced_example() -> Result<(), String> {
    // Load data from a file in one step
    let data = TrainingData::from_file("large_dataset.txt")?;
    println!("Vocabulary size: {}", data.vocab.len());

    // Configure advanced training parameters with fluent setters
    let config = TrainingConfig::new(ModelType::Cbow)
        .with_dim(500)
        .with_learning_rate(0.01)
        .with_epochs(50)
        .with_batch_size(128)
        .with_window(10)
        .with_negative_samples(10)
        .with_validation_ratio(0.2)
        .with_subsample_threshold(Some(1e-5))
        .with_unigram_negative_sampling(true)
        .with_warmup_epochs(Some(3))
        .with_checkpoint_interval(Some(10))
        .with_checkpoint_path(Some("./checkpoints".to_string()))
        .with_parallel(true);

    // Train model
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data)?;

    // Evaluate with cross-validation
    let cv = model.cross_validate(&data, 5)?;
    println!("Cross-validation accuracy: {:.4}", cv.averaged_metrics.accuracy);

    // Export to multiple formats
    model.save_embeddings("embeddings.txt", &data)?;
    println!("Training completed!");

    Ok(())
}
```

## 🔧 Configuration

### Training Parameters

| Parameter | Description | Default Value | Range |
|-----------|-------------|---------------|-------|
| `--dim` | Embedding dimension | 300 | 10-1000 |
| `--learning-rate` | Learning rate | 0.025 | 0.001-1.0 |
| `--epochs` | Number of training epochs | 10 | 1-1000 |
| `--batch-size` | Mini-batch size | 32 | 1-1000 |
| `--window` | Context window size | 5 | 1-20 |
| `--negative-samples` | Number of negative samples | 5 | 1-20 |
| `--validation-ratio` | Fraction of data for validation | 0.0 | 0.0-0.5 |
| `--validation-output` | File to write validation metrics JSON | - | - |

### Algorithm Types

- **`skipgram`**: Skip-gram algorithm (default)
- **`cbow`**: Continuous Bag of Words

### Export Formats

- **`text`**: Plain text format (default)
- **`json`**: JSON format with metadata
- **`bin`**: Binary format using bincode
- **`word2vec`**: Word2Vec/Gensim text format
- **`onnx-int8`**: Quantized ONNX with INT8 weights (~4x smaller)
- **`onnx-fp16`**: Quantized ONNX with FP16 weights (~2x smaller)

## 📖 CLI Reference

### Training Command

```bash
embedding train [OPTIONS]
```

**Options:**
- `--input <FILE>` - Input text file (required)
- `--output <FILE>` - Output model file (required)
- `--embeddings <FILE>` - Embeddings output file (required)
- `--dim <SIZE>` - Embedding dimension (default: 300)
- `--learning-rate <RATE>` - Learning rate (default: 0.025)
- `--epochs <COUNT>` - Number of epochs (default: 10)
- `--batch-size <SIZE>` - Batch size (default: 32)
- `--window <SIZE>` - Context window size (default: 5)
- `--negative-samples <COUNT>` - Negative samples (default: 5)
- `--model-type <TYPE>` - Algorithm type (skipgram|cbow)
- `--validation-ratio <RATIO>` - Fraction for validation (default: 0.0)
- `--validation-output <FILE>` - Path to write validation metrics JSON

### Similarity Command

```bash
embedding similarity <WORD1> <WORD2> [OPTIONS]
```

**Options:**
- `--model <FILE>` - Model file (required)

### Info Command

```bash
embedding info [OPTIONS]
```

**Options:**
- `--model <FILE>` - Model file (required)

### Export Command

```bash
embedding export [OPTIONS]
```

**Options:**
- `--model <FILE>` - Model file (required)
- `--output <FILE>` - Output file (required)
- `--format <FORMAT>` - Export format (text|json|bin|word2vec|onnx-int8|onnx-fp16)

### Validate Command

```bash
embedding validate [OPTIONS]
```

**Options:**
- `--model <FILE>` - Model file (required)
- `--input <FILE>` - Validation text file (required)
- `--output <FILE>` - Output metrics JSON file (optional)

## 🔍 Examples

### Example 1: Basic Word Embeddings

```bash
# Create sample data
cat > animals.txt << EOF
cat meows loudly
dog barks loudly
bird sings beautifully
fish swims quietly
horse gallops fast
EOF

# Train embeddings
embedding train \
    --input animals.txt \
    --output animal_model.json \
    --embeddings animal_embeddings.txt \
    --dim 50 \
    --epochs 20 \
    --model-type skipgram

# Test similarity
embedding similarity cat dog \
    --model animal_model.json
```

### Example 2: Document Embeddings

```bash
# Prepare document data
cat > documents.txt << EOF
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Natural language processing deals with text and speech.
Computer vision enables computers to understand images.
EOF

# Train with CBOW and validation
embedding train \
    --input documents.txt \
    --output doc_model.json \
    --embeddings doc_embeddings.txt \
    --dim 100 \
    --epochs 15 \
    --model-type cbow \
    --validation-ratio 0.2
```

### Example 3: Large Dataset Processing

```bash
# Process large file with multiple epochs
embedding train \
    --input large_corpus.txt \
    --output large_model.json \
    --embeddings large_embeddings.txt \
    --dim 300 \
    --epochs 50 \
    --batch-size 256 \
    --window 10 \
    --model-type cbow
```

## 🧪 Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/yingkitw/embedding.git
cd embedding

# Build development version
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build documentation
cargo doc --open
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_build_vocab

# Run with verbose output
cargo test -- --verbose
```

### Development Features

- **Unit Tests**: Comprehensive test coverage
- **Integration Tests**: End-to-end testing
- **Benchmarks**: Performance testing
- **Documentation**: API documentation

## 📊 Performance

### Benchmarks

| Algorithm | Vocab Size | Embed Dim | Training Time | Memory Usage |
|-----------|------------|-----------|---------------|--------------|
| Skip-gram | 10K words  | 300       | 2.3s          | 45MB         |
| CBOW      | 10K words  | 300       | 1.8s          | 42MB         |

### Optimization Tips

1. **Use appropriate batch sizes** for your dataset
2. **Adjust learning rate** based on dataset size
3. **Context window size** affects training speed and quality
4. **Use negative sampling** for large vocabularies
5. **Monitor memory usage** with large datasets

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Style

- Follow Rust formatting standards
- Use `cargo fmt` for code formatting
- Add comprehensive documentation
- Include tests for new features

## 📈 Roadmap

### Version 1.0 (Current)
- ✅ Skip-gram and CBOW algorithms
- ✅ CLI interface with train, validate, similarity, info, export
- ✅ Model persistence (JSON, binary, Word2Vec, ONNX, NumPy)
- ✅ Similarity calculations and semantic search
- ✅ Validation split and evaluation metrics (accuracy, precision, recall, F1)
- ✅ Learning rate scheduling (constant, exponential, step, cosine)
- ✅ Early stopping and L2 regularization

### Version 1.1 (Current — Features Complete)
- ✅ Backend abstraction trait for GPU acceleration (CPU implemented)
- ✅ WordPiece subword tokenization
- ✅ K-fold cross-validation support
- ✅ Per-epoch training history / learning curve JSON export
- ✅ Standard word similarity benchmark evaluation (Spearman correlation)
- ✅ K-means clustering
- CUDA/OpenCL backend implementation (planned)

### Version 2.0 (Current — Features Complete)
- ✅ Transformer encoder with multi-head self-attention and position encoding
- ✅ Enhanced multi-modal fusion (attention fusion, projection fusion, cross-modal similarity)
- ✅ Real-time incremental training (`IncrementalTrainer` with batch and stream modes)

### Version 2.1 (Current — Training Improvements Complete)
- ✅ Unigram^0.75 negative sampling distribution (Mikolov et al.)
- ✅ Sub-sampling of frequent words (`P(w) = 1 - sqrt(t / f(w))`)
- ✅ Learning rate warm-up (linear for first N epochs)
- ✅ Model checkpointing (save/resume every N epochs)
- ✅ Multi-threaded parallel training via `rayon`

## � Comparison with Alternatives

| Feature | **embedding** (this crate) | Gensim (Python) | rust-bert | fastText |
|---|---|---|---|---|
| **Language** | Rust | Python | Rust | C++ / Python |
| **Algorithms** | Skip-gram, CBOW, Transformer | Word2Vec, FastText, GloVe, LSI, LDA | BERT, RoBERTa, DistilBERT | Skip-gram, CBOW + subwords |
| **WordPiece tokenization** | ✅ | ❌ | ✅ | ❌ |
| **BPE tokenization** | ✅ | ❌ | ✅ | ❌ |
| **GPU acceleration** | ✅ (wgpu compute shaders, optional) | ❌ | ✅ (via ONNX / tch) | ✅ |
| **Cross-validation** | ✅ (k-fold) | ❌ | ❌ | ❌ |
| **Learning curves** | ✅ (per-epoch JSON export) | ❌ | ❌ | ❌ |
| **LR warm-up** | ✅ (linear epochs) | ❌ | ❌ | ❌ |
| **Checkpointing** | ✅ (save/resume) | ❌ | ❌ | ❌ |
| **Parallel training** | ✅ (rayon, multi-core) | ❌ | ❌ | ❌ |
| **Benchmark evaluation** | ✅ (Spearman correlation) | ✅ (similarity tasks) | ❌ | ✅ |
| **K-means clustering** | ✅ | ❌ | ❌ | ❌ |
| **Incremental training** | ✅ (stream / batch updates) | ❌ (requires retrain) | ❌ | ❌ |
| **Multi-modal fusion** | ✅ (4 fusion strategies) | ❌ | ❌ | ❌ |
| **CLI tool** | ✅ (train, validate, search, export) | ❌ | ❌ | ✅ |
| **Export formats** | JSON, binary, Word2Vec, ONNX, NumPy | Word2Vec, Gensim native | ONNX | `.vec`, `.bin` |
| **Memory mapping** | ✅ (binary format) | ✅ | ❌ | ✅ |
| **Pre-trained models** | ✅ (Word2Vec text/binary, GloVe, fastText, mmap .bin) | ✅ (many built-in) | ✅ (Hugging Face) | ✅ |
| **Sentence embeddings** | ✅ (mean pooling) | ✅ (Doc2Vec) | ✅ (BERT pooling) | ❌ |
| **Speed** | ⚡ Fast (Rust native) | 🐌 Python overhead | ⚡ Fast (Rust native) | ⚡ Fast (C++) |
| **Zero dependencies for inference** | ✅ (after training) | ❌ (Gensim + NumPy + SciPy) | ❌ (ONNX / torch) | ✅ (`.vec` format) |

> **Legend**: ✅ = Supported | ❌ = Not supported | 🔶 = Partial / planned

## �� Troubleshooting

### Common Issues

1. **Memory Error with Large Datasets**
   - Reduce batch size
   - Use streaming processing
   - Increase system memory

2. **Poor Similarity Results**
   - Increase training epochs
   - Adjust learning rate
   - Try different algorithms

3. **Missing Words in Vocabulary**
   - Check text preprocessing
   - Verify tokenization
   - Ensure words appear in text

### Performance Issues

- **Slow Training**: Reduce batch size or use negative sampling
- **High Memory Usage**: Use smaller embedding dimensions
- **Poor Quality**: Increase epochs or adjust parameters

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Word2Vec, GloVe, and BERT
- Built with [ndarray](https://github.com/rust-ndarray/ndarray) for numerical computing
- CLI powered by [clap](https://github.com/clap-rs/clap)
- Serialization using [serde](https://serde.rs/)