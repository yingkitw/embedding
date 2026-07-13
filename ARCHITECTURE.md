# Architecture - Embedding Trainer

## Module Structure

The codebase is organized into domain modules under `src/`:

| Module | File | Responsibility |
|--------|------|----------------|
| `lib` | `src/lib.rs` | Crate root: re-exports, ONNX protobuf, tests |
| `model` | `src/model.rs` | `EmbeddingModel` struct, constructors, query/similarity/analogy/evaluation methods |
| `training` | `src/training.rs` | Training algorithms (SkipGram, CBOW), gradient computation, learning rate scheduling with warm-up, early stopping, sub-sampling, negative sampling distribution, checkpointing, parallel training via `rayon` |
| `export` | `src/export.rs` | Export formats: text, Word2Vec, NumPy `.npy`, ONNX, memory-mapped binary, model checkpoints |
| `config` | `src/config.rs` | Training hyperparameters (`TrainingConfig`), `ModelType`, `TrainingData` with word frequencies |
| `code` | `src/code.rs` | Source code preprocessing (`CodeProcessor`), comment stripping, camelCase splitting, code tokenization (`load_code_data`) |
| `text` | `src/text.rs` | Text processing pipeline (`TextProcessor`), vocabulary building (`build_vocab`, `build_vocab_with_freq`), text loading |
| `tokenizer` | `src/tokenizer.rs` | Byte Pair Encoding tokenizer (`BPETokenizer`) |
| `search` | `src/search.rs` | Approximate nearest-neighbor search (`LSHIndex`), query expansion (`QueryExpander`), hierarchical clustering (`HierarchicalClustering`), k-means clustering |
| `hnsw` | `src/hnsw.rs` | HNSW graph index (`HNSWIndex`) for higher-recall approximate nearest neighbors |
| `benchmark` | `src/benchmark.rs` | Word similarity benchmark evaluation (`BenchmarkEvaluator`) with built-in WordSim-353, SimLex-999, MEN, RW, SCWS datasets |
| `quantization` | `src/quantization.rs` | Post-training INT8/FP16 quantization and quantized ONNX export |
| `evaluation` | `src/evaluation.rs` | Evaluation metrics (`EvaluationMetrics`, `ValidationData`, `TrainingHistory`, `CrossValidationResult`) |
| `transfer` | `src/transfer.rs` | Transfer learning: `MultimodalFusion`, `CrossLingualAligner`, `DomainAdapter`, `DocumentEmbedder`, `SubwordEmbedder`, `ZeroShotTransfer`, `IncrementalTrainer` |
| `mmap` | `src/mmap.rs` | Memory-mapped embedding storage (`MmapEmbeddings`) for read-only access to large binary files |
| `pretrained` | `src/pretrained.rs` | Pre-trained model loading: Word2Vec text/binary, GloVe, fastText, auto-detection |
| `cli` | `src/cli.rs` | CLI definitions (`Cli`, `Commands`, `run` dispatcher) |
| `commands` | `src/commands.rs` | CLI command handlers (`handle_train`, `handle_similarity`, `handle_info`, `handle_export`, `handle_validate`, `handle_interactive`) |

## Public API Surface

All public types are re-exported from `src/lib.rs` via `pub use <module>::*`, so consumers can import everything directly from the crate root:

```rust
use embedding::{EmbeddingModel, TrainingConfig, TextProcessor, HNSWIndex, BenchmarkEvaluator};
```

## Key Design Decisions

- **Domain modules over technical layers**: Code is grouped by business capability (training, text processing, search, evaluation, transfer) rather than by layer (models, services, repositories).
- **Thin crate root**: `lib.rs` focuses on module declarations, re-exports, and tests; all implementation lives in dedicated modules.
- **Tests in crate root**: The `#[cfg(test)]` module stays in `lib.rs` so it can exercise the full public API and any `pub(crate)` helpers without extra visibility changes.
- **Impl blocks across modules**: `EmbeddingModel` methods are split across `model.rs` (public API), `training.rs` (training internals), and `export.rs` (serialization) using multiple `impl` blocks, keeping each module focused on a single responsibility.
- **Validation integrated into training pipeline**: `handle_train` performs automatic train/validation splitting when `validation_ratio` is configured, then evaluates the trained model and reports metrics before saving.
- **Word frequency tracking**: `build_vocab_with_freq` captures word counts during vocabulary construction, enabling unigram-based negative sampling and sub-sampling without a second pass.
- **Parallel training via thread-local accumulators**: `epoch_skipgram_parallel` and `epoch_cbow_parallel` process sentences in parallel with per-thread `StdRng` and gradient accumulators, merging results at epoch end to avoid lock contention.
- **Checkpoint serialization**: Full model state is serialized as JSON (via serde), including embeddings and config, enabling exact resume from any saved epoch.
