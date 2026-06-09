# Architecture - Embedding Trainer

## Module Structure

The codebase is organized into domain modules under `src/`:

| Module | File | Responsibility |
|--------|------|----------------|
| `lib` | `src/lib.rs` | Crate root: re-exports, ONNX protobuf, `EmbeddingModel` core type, training/evaluation/search methods, and tests |
| `config` | `src/config.rs` | Training hyperparameters (`TrainingConfig`, `LearningRateSchedule`, `EarlyStoppingConfig`), `ModelType`, `TrainingData`, `DataLoader` |
| `text` | `src/text.rs` | Text processing pipeline (`TextProcessor`, `BPETokenizer`), vocabulary building (`build_vocab`), text loading (`load_text_data`) |
| `search` | `src/search.rs` | Approximate nearest-neighbor search (`LSHIndex`), query expansion (`QueryExpander`), hierarchical clustering (`HierarchicalClustering`) |
| `evaluation` | `src/evaluation.rs` | Evaluation metrics (`EvaluationMetrics`, `ValidationData`) |
| `transfer` | `src/transfer.rs` | Transfer learning and auxiliary structures: `MultimodalFusion`, `CrossLingualAligner`, `DomainAdapter`, `DocumentEmbedder`, `SubwordEmbedder`, `ZeroShotTransfer` |

## Public API Surface

All public types are re-exported from `src/lib.rs` via `pub use <module>::*`, so consumers can import everything directly from the crate root:

```rust
use embedding::{EmbeddingModel, TrainingConfig, TextProcessor, LSHIndex};
```

## Key Design Decisions

- **Domain modules over technical layers**: Code is grouped by business capability (training, text processing, search, evaluation, transfer) rather than by layer (models, services, repositories).
- **Thin crate root**: `lib.rs` focuses on the `EmbeddingModel` type and its intrinsic methods; auxiliary utilities live in their own modules.
- **Tests in crate root**: The `#[cfg(test)]` module stays in `lib.rs` so it can exercise the full public API and any `pub(crate)` helpers without extra visibility changes.

## Future Refactoring Notes

- `EmbeddingModel` and its training/evaluation/search methods could be extracted to a dedicated `model.rs` module if `lib.rs` grows further.
