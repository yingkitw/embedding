# Architecture - Embedding Trainer

## Module Structure

The codebase is organized into domain modules under `src/`:

| Module | File | Responsibility |
|--------|------|----------------|
| `lib` | `src/lib.rs` | Crate root: re-exports, ONNX protobuf, tests |
| `model` | `src/model.rs` | `EmbeddingModel` struct, constructors, query/similarity/analogy/evaluation methods |
| `training` | `src/training.rs` | Training algorithms (SkipGram, CBOW), gradient computation, learning rate scheduling, early stopping |
| `export` | `src/export.rs` | Export formats: text, Word2Vec, NumPy `.npy`, ONNX |
| `config` | `src/config.rs` | Training hyperparameters (`TrainingConfig`, `LearningRateSchedule`, `EarlyStoppingConfig`), `ModelType`, `TrainingData`, `DataLoader` |
| `code` | `src/code.rs` | Source code preprocessing (`CodeProcessor`), comment stripping, camelCase splitting, code tokenization (`load_code_data`) |
| `text` | `src/text.rs` | Text processing pipeline (`TextProcessor`), vocabulary building (`build_vocab`), text loading (`load_text_data`) |
| `tokenizer` | `src/tokenizer.rs` | Byte Pair Encoding tokenizer (`BPETokenizer`) |
| `search` | `src/search.rs` | Approximate nearest-neighbor search (`LSHIndex`), query expansion (`QueryExpander`), hierarchical clustering (`HierarchicalClustering`) |
| `evaluation` | `src/evaluation.rs` | Evaluation metrics (`EvaluationMetrics`, `ValidationData`) and model quality scoring |
| `transfer` | `src/transfer.rs` | Transfer learning and auxiliary structures: `MultimodalFusion`, `CrossLingualAligner`, `DomainAdapter`, `DocumentEmbedder`, `SubwordEmbedder`, `ZeroShotTransfer` |
| `cli` | `src/cli.rs` | CLI definitions (`Cli`, `Commands`, `run` dispatcher) |
| `commands` | `src/commands.rs` | CLI command handlers (`handle_train`, `handle_similarity`, `handle_info`, `handle_export`, `handle_validate`, `handle_interactive`) |

## Public API Surface

All public types are re-exported from `src/lib.rs` via `pub use <module>::*`, so consumers can import everything directly from the crate root:

```rust
use embedding::{EmbeddingModel, TrainingConfig, TextProcessor, LSHIndex};
```

## Key Design Decisions

- **Domain modules over technical layers**: Code is grouped by business capability (training, text processing, search, evaluation, transfer) rather than by layer (models, services, repositories).
- **Thin crate root**: `lib.rs` focuses on module declarations, re-exports, and tests; all implementation lives in dedicated modules.
- **Tests in crate root**: The `#[cfg(test)]` module stays in `lib.rs` so it can exercise the full public API and any `pub(crate)` helpers without extra visibility changes.
- **Impl blocks across modules**: `EmbeddingModel` methods are split across `model.rs` (public API), `training.rs` (training internals), and `export.rs` (serialization) using multiple `impl` blocks, keeping each module focused on a single responsibility.
- **Validation integrated into training pipeline**: `handle_train` performs automatic train/validation splitting when `validation_ratio` is configured, then evaluates the trained model and reports metrics before saving.
