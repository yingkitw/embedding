/// Evaluation metrics for a trained embedding model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub mean_similarity: f32,
    pub embedding_quality_score: f32,
}

/// Synthetic validation data generated from sentences for evaluation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ValidationData {
    pub positive_pairs: Vec<(String, String)>,
    pub negative_pairs: Vec<(String, String)>,
    pub analogies: Vec<(String, String, String, String)>,  // (word1, word2, word3, word4) for word1 - word2 + word3 = word4
}

/// Result of k-fold cross-validation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CrossValidationResult {
    pub folds: usize,
    pub averaged_metrics: EvaluationMetrics,
    pub per_fold_metrics: Vec<EvaluationMetrics>,
}

/// Metrics recorded for a single training epoch.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub loss: f32,
    pub learning_rate: f64,
    pub validation_metrics: Option<EvaluationMetrics>,
}

/// Accumulates per-epoch metrics to form a learning curve.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TrainingHistory {
    pub epochs: Vec<EpochMetrics>,
}

impl TrainingHistory {
    /// Creates an empty training history.
    pub fn new() -> Self {
        Self { epochs: Vec::new() }
    }

    /// Records metrics for a completed epoch.
    pub fn record_epoch(&mut self, epoch: usize, loss: f32, learning_rate: f64) {
        self.epochs.push(EpochMetrics {
            epoch,
            loss,
            learning_rate,
            validation_metrics: None,
        });
    }

    /// Records validation metrics for the most recent epoch.
    pub fn record_validation(&mut self, metrics: EvaluationMetrics) {
        if let Some(last) = self.epochs.last_mut() {
            last.validation_metrics = Some(metrics);
        }
    }

    /// Exports the full history as pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// Returns the average loss across all recorded epochs.
    pub fn average_loss(&self) -> f32 {
        if self.epochs.is_empty() {
            0.0
        } else {
            self.epochs.iter().map(|e| e.loss).sum::<f32>() / self.epochs.len() as f32
        }
    }

    /// Returns the final epoch's loss.
    pub fn final_loss(&self) -> f32 {
        self.epochs.last().map(|e| e.loss).unwrap_or(0.0)
    }
}
