use ndarray::{Array, Array1, Array2};
use rand::Rng;
use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};
use crate::config::{TrainingConfig, TrainingData, ModelType};
use crate::evaluation::{CrossValidationResult, EvaluationMetrics, TrainingHistory, ValidationData};
use crate::mmap;

/// Word embedding model with trained vector representations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    #[serde(with = "embeddings_serializer")]
    pub embeddings: Array2<f32>,
    pub config: TrainingConfig,
    pub vocab_size: usize,
    #[serde(skip)]
    pub training_history: TrainingHistory,
}

mod embeddings_serializer {
    use super::*;
    use serde::{Serialize, Deserialize};

    pub fn serialize<S>(embeddings: &Array2<f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let vec_vec: Vec<Vec<f32>> = embeddings.rows().into_iter()
            .map(|row| row.to_vec())
            .collect();
        vec_vec.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Array2<f32>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec_vec: Vec<Vec<f32>> = Vec::deserialize(deserializer)?;
        let rows = vec_vec.len();
        if rows == 0 {
            return Ok(Array::zeros((0, 0)));
        }
        let cols = vec_vec[0].len();
        
        let mut data = Vec::with_capacity(rows * cols);
        for row in vec_vec {
            data.extend_from_slice(&row);
        }
        
        Array::from_shape_vec((rows, cols), data)
            .map_err(|serde_err| <D::Error as serde::de::Error>::custom(format!("Invalid array shape: {}", serde_err)))
    }
}

impl EmbeddingModel {
    /// Creates a new embedding model with Xavier-initialized weights.
    pub fn new(config: TrainingConfig, vocab_size: usize) -> Self {
        let backend = crate::backend::best_backend();
        let embeddings = backend.init_embeddings(vocab_size, config.embedding_dim);

        Self {
            embeddings,
            config,
            vocab_size,
            training_history: TrainingHistory::new(),
        }
    }

    /// Creates a new embedding model initialized from a pre-trained Word2Vec file.
    pub fn new_with_pretrained(
        config: TrainingConfig,
        vocab_size: usize,
        data: &TrainingData,
        pretrained_path: &str,
    ) -> Result<Self, String> {
        let (pretrained, pretrained_dim) = Self::load_word2vec_format(pretrained_path)?;
        if pretrained_dim != config.embedding_dim {
            return Err(format!(
                "Pre-trained embedding dimension ({}) does not match config ({})",
                pretrained_dim, config.embedding_dim
            ));
        }

        let mut rng = rand::thread_rng();
        let scale = 1.0 / (config.embedding_dim as f32).sqrt();
        let mut embeddings = Array::from_shape_fn((vocab_size, config.embedding_dim), |_| {
            rng.gen_range(-0.5..0.5) * scale
        });

        let mut loaded_count = 0;
        for (word, word_id) in &data.vocab {
            if let Some(pretrained_vec) = pretrained.get(word) {
                for (i, &val) in pretrained_vec.iter().enumerate() {
                    embeddings[[*word_id, i]] = val;
                }
                loaded_count += 1;
            }
        }

        tracing::info!(
            "Loaded {} pre-trained embeddings out of {} vocabulary words",
            loaded_count,
            vocab_size
        );

        Ok(Self {
            embeddings,
            config,
            vocab_size,
            training_history: TrainingHistory::new(),
        })
    }

    /// Trains the model on the provided data.
    pub fn train(&mut self, data: &TrainingData) -> Result<(), String> {
        match self.config.model_type {
            ModelType::SkipGram => self.train_skipgram(data),
            ModelType::Cbow => self.train_cbow(data),
        }
    }

    /// Returns the embedding vector for a given word.
    pub fn get_embedding(&self, word: &str, data: &TrainingData) -> Option<Array1<f32>> {
        if let Some(&word_id) = data.vocab.get(word) {
            if word_id < self.embeddings.nrows() {
                Some(self.embeddings.row(word_id).to_owned())
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Returns the top-k most similar words to the query using full cosine similarity.
    pub fn semantic_search(&self, query: &str, data: &TrainingData, top_k: usize) -> Vec<(String, f32)> {
        let query_emb = match self.get_embedding(query, data) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut results = Vec::new();
        for (word_id, word) in data.reverse_vocab.iter().enumerate() {
            if word == query {
                continue;
            }
            let candidate = self.embeddings.row(word_id);
            let dot: f32 = query_emb.iter().zip(candidate.iter()).map(|(&a, &b)| a * b).sum();
            let norm_query = query_emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm_candidate = candidate.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_query > 0.0 && norm_candidate > 0.0 {
                results.push((word.clone(), dot / (norm_query * norm_candidate)));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_k).collect()
    }

    /// Computes the vector difference `emb(word1) - emb(word2)`.
    pub fn embedding_arithmetic(&self, word1: &str, word2: &str, data: &TrainingData) -> Option<Array1<f32>> {
        let emb1 = self.get_embedding(word1, data)?;
        let emb2 = self.get_embedding(word2, data)?;
        Some(&emb1 - &emb2)
    }

    /// Linearly interpolates between two word embeddings.
    pub fn interpolate_embeddings(&self, word1: &str, word2: &str, data: &TrainingData, alpha: f32) -> Option<Array1<f32>> {
        let emb1 = self.get_embedding(word1, data)?;
        let emb2 = self.get_embedding(word2, data)?;
        Some(&emb1 * alpha + &emb2 * (1.0 - alpha))
    }

    /// Cosine similarity between two words in the learned embedding space.
    pub fn similarity(&self, word1: &str, word2: &str, data: &TrainingData) -> Option<f32> {
        let emb1 = self.get_embedding(word1, data)?;
        let emb2 = self.get_embedding(word2, data)?;
        
        let dot_product = emb1.iter().zip(emb2.iter()).map(|(&a, &b)| a * b).sum::<f32>();
        let norm1 = emb1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2 = emb2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            None
        } else {
            Some(dot_product / (norm1 * norm2))
        }
    }
    
    /// Evaluates the model against synthetic validation pairs and analogies.
    pub fn evaluate(&self, data: &TrainingData, validation_data: &ValidationData) -> EvaluationMetrics {
        let threshold = 0.5f32;
        let mut pos_sims = Vec::new();
        let mut neg_sims = Vec::new();

        for (word1, word2) in validation_data.positive_pairs.iter() {
            if let Some(sim) = self.similarity(word1, word2, data) {
                pos_sims.push(sim);
            }
        }

        for (word1, word2) in validation_data.negative_pairs.iter() {
            if let Some(sim) = self.similarity(word1, word2, data) {
                neg_sims.push(sim);
            }
        }

        let mut correct = 0usize;
        let mut total = 0usize;

        for &sim in &pos_sims {
            total += 1;
            if sim >= threshold { correct += 1; }
        }

        for &sim in &neg_sims {
            total += 1;
            if sim < threshold { correct += 1; }
        }

        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

        let mean_pos = if !pos_sims.is_empty() { pos_sims.iter().sum::<f32>() / pos_sims.len() as f32 } else { 0.0 };
        let mean_neg = if !neg_sims.is_empty() { neg_sims.iter().sum::<f32>() / neg_sims.len() as f32 } else { 0.0 };
        let mean_similarity = (mean_pos + mean_neg) / 2.0;

        // F1 score: treat positive pairs as "positive class"
        let tp = pos_sims.iter().filter(|&&s| s >= threshold).count() as f32;
        let fp = neg_sims.iter().filter(|&&s| s >= threshold).count() as f32;
        let fn_ = pos_sims.iter().filter(|&&s| s < threshold).count() as f32;

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

        let embedding_quality_score = self.calculate_embedding_quality(data);

        EvaluationMetrics {
            accuracy,
            precision,
            recall,
            f1_score: f1,
            mean_similarity,
            embedding_quality_score,
        }
    }
    
    fn calculate_embedding_quality(&self, _data: &TrainingData) -> f32 {
        let mut total_norm = 0.0;
        let mut count = 0;
        let mut total_variance = 0.0;
        let vocab_size = self.embeddings.nrows();

        for word_id in 0..vocab_size {
            let embedding = self.embeddings.row(word_id);
            let norm = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
            total_norm += norm;
            count += 1;

            // Calculate variance within embedding dimensions
            let mean_val = embedding.sum() / self.config.embedding_dim as f32;
            let variance = embedding.iter().map(|&x| (x - mean_val).powi(2)).sum::<f32>() / self.config.embedding_dim as f32;
            total_variance += variance;
        }

        let avg_norm = if count > 0 { total_norm / count as f32 } else { 0.0 };
        let avg_variance = if count > 0 { total_variance / count as f32 } else { 0.0 };

        // Quality score based on norm and variance (higher is better)
        let quality = (avg_norm * avg_variance).sqrt();
        quality.min(1.0)  // Normalize to 0-1
    }

    /// L2-normalizes every embedding vector to unit length.
    pub fn normalize_embeddings(&mut self) {
        for mut row in self.embeddings.rows_mut() {
            let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.map_inplace(|x| *x /= norm);
            }
        }
    }

    /// Solves word analogies of the form `word1 : word2 :: word3 : ?`.
    pub fn analogy(&self, word1: &str, word2: &str, word3: &str, data: &TrainingData, top_k: usize) -> Vec<(String, f32)> {
        let emb1 = match self.get_embedding(word1, data) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let emb2 = match self.get_embedding(word2, data) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let emb3 = match self.get_embedding(word3, data) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let target = &emb3 + &emb1 - &emb2;
        let mut results = Vec::new();

        for (word_id, word) in data.reverse_vocab.iter().enumerate() {
            if word == word1 || word == word2 || word == word3 {
                continue;
            }
            let candidate = self.embeddings.row(word_id);
            let dot: f32 = target.iter().zip(candidate.iter()).map(|(&a, &b)| a * b).sum();
            let norm_target = target.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm_candidate = candidate.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_target > 0.0 && norm_candidate > 0.0 {
                results.push((word.clone(), dot / (norm_target * norm_candidate)));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_k).collect()
    }

    /// Splits sentences into training and validation sets.
    pub fn split_data(&self, sentences: &[Vec<String>], train_ratio: f64) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
        let total_sentences = sentences.len();
        let train_size = (total_sentences as f64 * train_ratio) as usize;
        
        let mut shuffled_indices: Vec<usize> = (0..total_sentences).collect();
        let mut rng = rand::thread_rng();
        shuffled_indices.shuffle(&mut rng);
        
        let train_sentences: Vec<Vec<String>> = shuffled_indices[..train_size]
            .iter()
            .map(|&i| sentences[i].clone())
            .collect();
        
        let val_sentences: Vec<Vec<String>> = shuffled_indices[train_size..]
            .iter()
            .map(|&i| sentences[i].clone())
            .collect();
        
        (train_sentences, val_sentences)
    }
    
    /// Generates synthetic validation pairs and analogies from sentences.
    pub fn create_validation_data(&self, sentences: &[Vec<String>]) -> ValidationData {
        let mut positive_pairs = Vec::new();
        let mut negative_pairs = Vec::new();
        let mut analogies = Vec::new();
        
        // Simple heuristic to create word pairs
        for sentence in sentences {
            if sentence.len() >= 2 {
                // Consecutive words as positive pairs
                for i in 0..sentence.len() - 1 {
                    positive_pairs.push((sentence[i].clone(), sentence[i + 1].clone()));
                }
                
                // Non-consecutive words as negative pairs
                if sentence.len() >= 3 {
                    for i in 0..sentence.len() - 2 {
                        negative_pairs.push((sentence[i].clone(), sentence[i + 2].clone()));
                    }
                }
            }
        }
        
        // Simple analogies (this is a simplified version)
        if sentences.len() >= 4 {
            for i in 0..std::cmp::min(10, sentences.len() - 3) {
                let s1 = &sentences[i];
                let s2 = &sentences[i + 1];
                let s3 = &sentences[i + 2];
                let s4 = &sentences[i + 3];
                
                if !s1.is_empty() && !s2.is_empty() && !s3.is_empty() && !s4.is_empty() {
                    analogies.push((
                        s1[0].clone(),
                        s2[0].clone(),
                        s3[0].clone(),
                        s4[0].clone(),
                    ));
                }
            }
        }
        
        ValidationData {
            positive_pairs,
            negative_pairs,
            analogies,
        }
    }

    /// Expands the vocabulary and embedding matrix with new words.
    pub fn incremental_vocab_update(
        &mut self,
        new_words: &[String],
        data: &mut TrainingData,
    ) -> Result<Vec<usize>, String> {
        let mut added_ids = Vec::new();
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (self.config.embedding_dim as f32).sqrt();

        for word in new_words {
            if data.vocab.contains_key(word) {
                continue;
            }
            let new_id = data.vocab.len();
            data.vocab.insert(word.clone(), new_id);
            data.reverse_vocab.push(word.clone());
            added_ids.push(new_id);
        }

        if added_ids.is_empty() {
            return Ok(added_ids);
        }

        // Expand embeddings matrix with Xavier initialization for new words
        let new_size = data.vocab.len();
        let mut new_embeddings = Array::from_shape_fn((new_size, self.config.embedding_dim), |_| {
            rng.gen_range(-0.5..0.5) * scale
        });

        // Copy old embeddings
        for i in 0..self.vocab_size {
            for j in 0..self.config.embedding_dim {
                new_embeddings[[i, j]] = self.embeddings[[i, j]];
            }
        }

        self.embeddings = new_embeddings;
        self.vocab_size = new_size;

        Ok(added_ids)
    }

    /// Computes a sentence-level embedding by mean-pooling word embeddings.
    pub fn sentence_embedding(&self, sentence: &[String], data: &TrainingData) -> Option<Array1<f32>> {
        if sentence.is_empty() {
            return None;
        }
        let mut sum = Array::zeros(self.config.embedding_dim);
        let mut count = 0usize;
        for word in sentence {
            if let Some(emb) = self.get_embedding(word, data) {
                sum += &emb;
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        Some(&sum / (count as f32))
    }

    /// Performs k-fold cross-validation on the given data.
    ///
    /// Splits sentences into `k` folds, trains a fresh model on k-1 folds,
    /// and evaluates on the held-out fold. Returns averaged metrics and
    /// per-fold results.
    pub fn cross_validate(
        &self,
        data: &TrainingData,
        k: usize,
    ) -> Result<CrossValidationResult, String> {
        if k < 2 || k > data.sentences.len() {
            return Err("k must be between 2 and the number of sentences".to_string());
        }

        let mut shuffled_indices: Vec<usize> = (0..data.sentences.len()).collect();
        let mut rng = rand::thread_rng();
        shuffled_indices.shuffle(&mut rng);

        let fold_size = data.sentences.len() / k;
        let mut per_fold_metrics = Vec::with_capacity(k);

        for fold in 0..k {
            let start = fold * fold_size;
            let end = if fold == k - 1 {
                data.sentences.len()
            } else {
                start + fold_size
            };

            let val_indices: std::collections::HashSet<usize> =
                shuffled_indices[start..end].iter().copied().collect();

            let train_sentences: Vec<Vec<String>> = shuffled_indices
                .iter()
                .filter(|&&i| !val_indices.contains(&i))
                .map(|&i| data.sentences[i].clone())
                .collect();

            let val_sentences: Vec<Vec<String>> = shuffled_indices[start..end]
                .iter()
                .map(|&i| data.sentences[i].clone())
                .collect();

            // Build training vocabulary from train sentences
            let (train_vocab, train_reverse) = crate::text::build_vocab(&train_sentences);
            let train_data = TrainingData {
                sentences: train_sentences,
                vocab: train_vocab,
                reverse_vocab: train_reverse,
            };

            // Train a fresh model
            let mut fold_model = EmbeddingModel::new(self.config.clone(), train_data.vocab.len());
            fold_model.train(&train_data)?;

            // Build validation data using the training vocab only
            // Words not in training vocab will be treated as OOV
            let val_data = TrainingData {
                sentences: val_sentences,
                vocab: train_data.vocab.clone(),
                reverse_vocab: train_data.reverse_vocab.clone(),
            };

            let validation_pairs = fold_model.create_validation_data(&val_data.sentences);
            let metrics = fold_model.evaluate(&val_data, &validation_pairs);
            per_fold_metrics.push(metrics);
        }

        let n = per_fold_metrics.len() as f32;
        let avg = EvaluationMetrics {
            accuracy: per_fold_metrics.iter().map(|m| m.accuracy).sum::<f32>() / n,
            precision: per_fold_metrics.iter().map(|m| m.precision).sum::<f32>() / n,
            recall: per_fold_metrics.iter().map(|m| m.recall).sum::<f32>() / n,
            f1_score: per_fold_metrics.iter().map(|m| m.f1_score).sum::<f32>() / n,
            mean_similarity: per_fold_metrics.iter().map(|m| m.mean_similarity).sum::<f32>() / n,
            embedding_quality_score: per_fold_metrics
                .iter()
                .map(|m| m.embedding_quality_score)
                .sum::<f32>()
                / n,
        };

        Ok(CrossValidationResult {
            folds: k,
            averaged_metrics: avg,
            per_fold_metrics,
        })
    }
}
