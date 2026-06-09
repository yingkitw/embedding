/// Evaluation metrics for a trained embedding model.
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub mean_similarity: f32,
    pub embedding_quality_score: f32,
}

/// Synthetic validation data generated from sentences for evaluation.
#[derive(Debug, Clone)]
pub struct ValidationData {
    pub positive_pairs: Vec<(String, String)>,
    pub negative_pairs: Vec<(String, String)>,
    pub analogies: Vec<(String, String, String, String)>,  // (word1, word2, word3, word4) for word1 - word2 + word3 = word4
}
