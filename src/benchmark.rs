use crate::{EmbeddingModel, TrainingData};

/// A single word pair with a human-annotated similarity score.
#[derive(Debug, Clone)]
pub struct BenchmarkPair {
    pub word1: String,
    pub word2: String,
    pub score: f32,
}

/// Result of evaluating a model on a word similarity benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub num_pairs: usize,
    pub num_evaluated: usize,
    pub correlation: f32,
    pub model_scores: Vec<f32>,
    pub human_scores: Vec<f32>,
}

/// Evaluates embedding models against standard word similarity benchmarks
/// such as WordSim-353 or SimLex-999.
pub struct BenchmarkEvaluator;

impl BenchmarkEvaluator {
    /// Parses a TSV benchmark file where each line is `word1\tword2\tscore`.
    pub fn load_from_tsv(text: &str) -> Vec<BenchmarkPair> {
        let mut pairs = Vec::new();
        for line in text.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 && let Ok(score) = parts[2].parse::<f32>() {
                pairs.push(BenchmarkPair {
                    word1: parts[0].trim().to_lowercase(),
                    word2: parts[1].trim().to_lowercase(),
                    score,
                });
            }
        }
        pairs
    }

    /// Computes Spearman rank correlation between model cosine similarities
    /// and human similarity scores for a list of benchmark pairs.
    pub fn evaluate(
        model: &EmbeddingModel,
        data: &TrainingData,
        pairs: &[BenchmarkPair],
    ) -> BenchmarkResult {
        let mut model_scores = Vec::new();
        let mut human_scores = Vec::new();

        for pair in pairs {
            if let Some(sim) = model.similarity(&pair.word1, &pair.word2, data) {
                model_scores.push(sim);
                human_scores.push(pair.score);
            }
        }

        let correlation = if model_scores.len() >= 2 {
            spearman_correlation(&model_scores, &human_scores)
        } else {
            0.0
        };

        BenchmarkResult {
            num_pairs: pairs.len(),
            num_evaluated: model_scores.len(),
            correlation,
            model_scores,
            human_scores,
        }
    }
}

/// Computes the Spearman rank correlation coefficient between two vectors.
fn spearman_correlation(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f32;
    if n <= 1.0 {
        return 0.0;
    }

    let x_ranks = rank(x);
    let y_ranks = rank(y);

    let mean_x = x_ranks.iter().sum::<f32>() / n;
    let mean_y = y_ranks.iter().sum::<f32>() / n;

    let mut num = 0.0f32;
    let mut den_x = 0.0f32;
    let mut den_y = 0.0f32;

    for i in 0..x_ranks.len() {
        let dx = x_ranks[i] - mean_x;
        let dy = y_ranks[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    let den = (den_x * den_y).sqrt();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Assigns ranks to a vector (1-based, averaging ties).
fn rank(values: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < indexed.len() {
        let mut j = i;
        while j + 1 < indexed.len() && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-9 {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f32 / 2.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}
