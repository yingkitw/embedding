use ndarray::{Array, Array2, Array1};
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub embedding_dim: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub context_window: usize,
    pub negative_samples: usize,
    pub model_type: ModelType,
    pub lr_schedule: LearningRateSchedule,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub l2_regularization: Option<f64>,
    pub dropout_rate: Option<f32>,
    pub gradient_clip: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant,
    Exponential { decay_rate: f64 },
    Step { step_size: usize, gamma: f64 },
    Cosine { t_max: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    SkipGram,
    Cbow,
    SentenceBERT,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    pub sentences: Vec<Vec<String>>,
    pub vocab: HashMap<String, usize>,
    pub reverse_vocab: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    #[serde(with = "embeddings_serializer")]
    pub embeddings: Array2<f32>,
    pub config: TrainingConfig,
    pub vocab_size: usize,
    pub memory_mapped: bool,
}

#[derive(Debug, Clone)]
pub struct DataLoader {
    pub batch_size: usize,
    pub shuffle: bool,
    pub use_memory_mapping: bool,
    pub file_path: Option<String>,
}

impl DataLoader {
    pub fn new(batch_size: usize, shuffle: bool, use_memory_mapping: bool) -> Self {
        Self {
            batch_size,
            shuffle,
            use_memory_mapping,
            file_path: None,
        }
    }
    
    pub fn set_file_path(&mut self, path: String) {
        self.file_path = Some(path);
    }
    
    pub fn load_batches(&self, sentences: &[Vec<String>]) -> Vec<Vec<Vec<String>>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        
        for sentence in sentences {
            current_batch.push(sentence.clone());
            
            if current_batch.len() >= self.batch_size {
                if self.shuffle {
                    let mut rng = rand::thread_rng();
                    current_batch.shuffle(&mut rng);
                }
                batches.push(current_batch.clone());
                current_batch.clear();
            }
        }
        
        // Add remaining sentences as the last batch
        if !current_batch.is_empty() {
            if self.shuffle {
                let mut rng = rand::thread_rng();
                current_batch.shuffle(&mut rng);
            }
            batches.push(current_batch);
        }
        
        batches
    }
    
    pub fn load_lazily(&self, file_path: &str) -> Result<Vec<Vec<String>>, String> {
        if self.use_memory_mapping {
            self.load_with_memory_mapping(file_path)
        } else {
            self.load_regular(file_path)
        }
    }
    
    fn load_regular(&self, file_path: &str) -> Result<Vec<Vec<String>>, String> {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(file_path).map_err(|e| e.to_string())?;
        let mut content = String::new();
        file.read_to_string(&mut content).map_err(|e| e.to_string())?;
        
        Ok(load_text_data(&content))
    }
    
    fn load_with_memory_mapping(&self, file_path: &str) -> Result<Vec<Vec<String>>, String> {
        // For memory-mapped files, we'd typically use a library like memmap2
        // For now, we'll just simulate the behavior
        self.load_regular(file_path)
    }
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
        
        Ok(Array::from_shape_vec((rows, cols), data)
            .map_err(|serde_err| <D::Error as serde::de::Error>::custom(format!("Invalid array shape: {}", serde_err)))?)
    }
}

impl EmbeddingModel {
    pub fn new(config: TrainingConfig, vocab_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (config.embedding_dim as f32).sqrt();
        let embeddings = Array::from_shape_fn((vocab_size, config.embedding_dim), |_| {
            rng.gen_range(-0.5..0.5) * scale
        });

        Self {
            embeddings,
            config,
            vocab_size,
            memory_mapped: false,
        }
    }

    pub fn train(&mut self, data: &TrainingData) -> Result<(), String> {
        match self.config.model_type {
            ModelType::SkipGram => self.train_skipgram(data),
            ModelType::Cbow => self.train_cbow(data),
            ModelType::SentenceBERT => self.train_sentence_bert(data),
        }
    }

    fn train_skipgram(&mut self, data: &TrainingData) -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut best_loss = f32::MAX;
        let mut patience_counter = 0;

        for epoch in 0..self.config.epochs {
            info!("Epoch {}/{}", epoch + 1, self.config.epochs);

            let current_lr = self.get_learning_rate(epoch, self.config.epochs);
            info!("Current learning rate: {:.6}", current_lr);

            let mut total_loss = 0.0;
            let mut num_updates = 0;
            let mut batch_count = 0;
            let mut batch_loss = 0.0;
            let mut num_batches = 0;
            let mut accum = self.new_gradient_accumulator();

            for (_sentence_idx, sentence) in data.sentences.iter().enumerate() {
                for (target_idx, target_word) in sentence.iter().enumerate() {
                    if let Some(&target_id) = data.vocab.get(target_word) {
                        let start = target_idx.saturating_sub(self.config.context_window);
                        let end = std::cmp::min(target_idx + self.config.context_window + 1, sentence.len());

                        // Get negative samples
                        let negative_samples = get_negative_samples(
                            self.vocab_size,
                            self.config.negative_samples,
                            target_id,
                            &mut rng
                        );

                        for context_idx in start..end {
                            if context_idx != target_idx {
                                if let Some(&context_id) = data.vocab.get(&sentence[context_idx]) {
                                    let loss = self.compute_skipgram_gradients(
                                        target_id, context_id, &negative_samples, current_lr, &mut accum
                                    );
                                    total_loss += loss;
                                    batch_loss += loss;
                                    num_updates += 1;
                                    batch_count += 1;

                                    if batch_count >= self.config.batch_size {
                                        self.apply_gradient_batch(&mut accum, batch_count);
                                        num_batches += 1;
                                        if num_batches % 10 == 0 {
                                            info!("  Batch {} avg loss: {:.4}", num_batches, batch_loss / batch_count as f32);
                                        }
                                        batch_loss = 0.0;
                                        batch_count = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Apply any remaining gradients
            if batch_count > 0 {
                self.apply_gradient_batch(&mut accum, batch_count);
            }

            let avg_loss = total_loss / num_updates.max(1) as f32;
            info!("Epoch {} completed. Average loss: {:.4}", epoch + 1, avg_loss);

            // Early stopping check
            if self.should_early_stop(epoch + 1, avg_loss, best_loss, &mut patience_counter) {
                info!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }

            if avg_loss < best_loss {
                best_loss = avg_loss;
                patience_counter = 0;
            }
        }

        Ok(())
    }

    fn train_cbow(&mut self, data: &TrainingData) -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut best_loss = f32::MAX;
        let mut patience_counter = 0;

        for epoch in 0..self.config.epochs {
            info!("Epoch {}/{}", epoch + 1, self.config.epochs);

            let current_lr = self.get_learning_rate(epoch, self.config.epochs);
            info!("Current learning rate: {:.6}", current_lr);

            let mut total_loss = 0.0;
            let mut num_updates = 0;
            let mut batch_count = 0;
            let mut batch_loss = 0.0;
            let mut num_batches = 0;
            let mut accum = self.new_gradient_accumulator();

            for (_sentence_idx, sentence) in data.sentences.iter().enumerate() {
                for (target_idx, target_word) in sentence.iter().enumerate() {
                    if let Some(&target_id) = data.vocab.get(target_word) {
                        let start = target_idx.saturating_sub(self.config.context_window);
                        let end = std::cmp::min(target_idx + self.config.context_window + 1, sentence.len());

                        let mut context_ids = Vec::new();
                        for context_idx in start..end {
                            if context_idx != target_idx {
                                if let Some(&context_id) = data.vocab.get(&sentence[context_idx]) {
                                    context_ids.push(context_id);
                                }
                            }
                        }

                        if !context_ids.is_empty() {
                            // Get negative samples
                            let negative_samples = get_negative_samples(
                                self.vocab_size,
                                self.config.negative_samples,
                                target_id,
                                &mut rng
                            );

                            let loss = self.compute_cbow_gradients(
                                target_id, &context_ids, &negative_samples, current_lr, &mut accum
                            );
                            total_loss += loss;
                            batch_loss += loss;
                            num_updates += 1;
                            batch_count += 1;

                            if batch_count >= self.config.batch_size {
                                self.apply_gradient_batch(&mut accum, batch_count);
                                num_batches += 1;
                                if num_batches % 10 == 0 {
                                    info!("  Batch {} avg loss: {:.4}", num_batches, batch_loss / batch_count as f32);
                                }
                                batch_loss = 0.0;
                                batch_count = 0;
                            }
                        }
                    }
                }
            }

            // Apply any remaining gradients
            if batch_count > 0 {
                self.apply_gradient_batch(&mut accum, batch_count);
            }

            let avg_loss = total_loss / num_updates.max(1) as f32;
            info!("Epoch {} completed. Average loss: {:.4}", epoch + 1, avg_loss);

            // Early stopping check
            if self.should_early_stop(epoch + 1, avg_loss, best_loss, &mut patience_counter) {
                info!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }

            if avg_loss < best_loss {
                best_loss = avg_loss;
                patience_counter = 0;
            }
        }

        Ok(())
    }

    fn train_sentence_bert(&mut self, data: &TrainingData) -> Result<(), String> {
        info!("Training Sentence-BERT model (simplified implementation)");
        
        // For Sentence-BERT style training, we'll use a mean pooling approach
        for epoch in 0..self.config.epochs {
            info!("Epoch {}/{}", epoch + 1, self.config.epochs);
            
            for sentence in data.sentences.iter() {
                let word_ids: Vec<usize> = sentence.iter()
                    .filter_map(|word| data.vocab.get(word))
                    .copied()
                    .collect();
                
                if !word_ids.is_empty() {
                    // Simple mean pooling for sentence embeddings
                    let mean_embedding = self.mean_pooling(&word_ids);
                    self.update_embeddings_from_mean(mean_embedding, &word_ids);
                }
            }
        }
        
        Ok(())
    }

    fn mean_pooling(&self, word_ids: &[usize]) -> Array1<f32> {
        let mut sum = Array::zeros(self.config.embedding_dim);
        let mut count = 0;
        
        for &word_id in word_ids {
            if word_id < self.vocab_size {
                let embedding = self.embeddings.row(word_id);
                sum = sum + embedding;
                count += 1;
            }
        }
        
        if count > 0 {
            sum / count as f32
        } else {
            Array::zeros(self.config.embedding_dim)
        }
    }

    fn update_embeddings_from_mean(&mut self, mean_embedding: Array1<f32>, word_ids: &[usize]) {
        for &word_id in word_ids {
            if word_id < self.vocab_size {
                let mut word_embedding = self.embeddings.row_mut(word_id);
                // Move towards the mean embedding
                for i in 0..self.config.embedding_dim {
                    word_embedding[i] += 0.01 * (mean_embedding[i] - word_embedding[i]);
                }
            }
        }
    }

    fn new_gradient_accumulator(&self) -> HashMap<usize, Vec<f32>> {
        HashMap::new()
    }

    fn accumulate_gradient(&self, accum: &mut HashMap<usize, Vec<f32>>, word_id: usize, grad: f32, dim_idx: usize) {
        accum.entry(word_id)
            .or_insert_with(|| vec![0.0; self.config.embedding_dim])[dim_idx] += grad;
    }

    fn apply_gradient_batch(&mut self, accum: &mut HashMap<usize, Vec<f32>>, batch_size: usize) {
        let scale = 1.0 / batch_size.max(1) as f32;
        for (&word_id, grads) in accum.iter() {
            for (i, &grad) in grads.iter().enumerate() {
                self.embeddings[[word_id, i]] -= self.clip_gradient(grad * scale);
            }
        }
        accum.clear();
    }

    fn compute_skipgram_gradients(
        &self,
        target_id: usize,
        context_id: usize,
        negative_samples: &[usize],
        learning_rate: f32,
        accum: &mut HashMap<usize, Vec<f32>>,
    ) -> f32 {
        let target_embedding: Vec<f32> = self.embeddings.row(target_id).to_vec();
        let context_embedding: Vec<f32> = self.embeddings.row(context_id).to_vec();

        let dot_product: f32 = target_embedding.iter().zip(context_embedding.iter()).map(|(&a, &b)| a * b).sum();
        let prob_positive = sigmoid(dot_product);
        let grad_positive = prob_positive - 1.0;

        for i in 0..self.config.embedding_dim {
            let mut grad = learning_rate * grad_positive * context_embedding[i];
            if let Some(l2_reg) = self.config.l2_regularization {
                grad += learning_rate * l2_reg as f32 * target_embedding[i];
            }
            self.accumulate_gradient(accum, target_id, grad, i);

            let mut grad_context = learning_rate * grad_positive * target_embedding[i];
            if let Some(l2_reg) = self.config.l2_regularization {
                grad_context += learning_rate * l2_reg as f32 * context_embedding[i];
            }
            self.accumulate_gradient(accum, context_id, grad_context, i);
        }

        for &neg_id in negative_samples {
            let neg_embedding: Vec<f32> = self.embeddings.row(neg_id).to_vec();
            let dot_product_neg: f32 = target_embedding.iter().zip(neg_embedding.iter()).map(|(&a, &b)| a * b).sum();
            let prob_negative = sigmoid(dot_product_neg);
            let grad_negative = prob_negative;

            for i in 0..self.config.embedding_dim {
                let mut grad = learning_rate * grad_negative * neg_embedding[i];
                if let Some(l2_reg) = self.config.l2_regularization {
                    grad += learning_rate * l2_reg as f32 * target_embedding[i];
                }
                self.accumulate_gradient(accum, target_id, grad, i);

                let mut grad_neg = learning_rate * grad_negative * target_embedding[i];
                if let Some(l2_reg) = self.config.l2_regularization {
                    grad_neg += learning_rate * l2_reg as f32 * neg_embedding[i];
                }
                self.accumulate_gradient(accum, neg_id, grad_neg, i);
            }
        }

        let mut loss = -prob_positive.ln();
        for &neg_id in negative_samples {
            let neg_embedding: Vec<f32> = self.embeddings.row(neg_id).to_vec();
            let dot_product_neg: f32 = target_embedding.iter().zip(neg_embedding.iter()).map(|(&a, &b)| a * b).sum();
            loss += -sigmoid(-dot_product_neg).ln();
        }
        loss
    }

    fn compute_cbow_gradients(
        &self,
        target_id: usize,
        context_ids: &[usize],
        negative_samples: &[usize],
        learning_rate: f32,
        accum: &mut HashMap<usize, Vec<f32>>,
    ) -> f32 {
        let mut context_vector = Array::zeros(self.config.embedding_dim);
        let mut context_embeddings: Vec<Vec<f32>> = Vec::new();

        for &context_id in context_ids {
            let context_embedding: Vec<f32> = self.embeddings.row(context_id).to_vec();
            context_embeddings.push(context_embedding.clone());
            let context_arr = Array::from_shape_vec((self.config.embedding_dim,), context_embedding).unwrap();
            context_vector = context_vector + &context_arr;
        }
        context_vector = context_vector / context_ids.len() as f32;

        let target_embedding: Vec<f32> = self.embeddings.row(target_id).to_vec();
        let target_arr = Array::from_shape_vec((self.config.embedding_dim,), target_embedding.clone()).unwrap();
        let dot_product = context_vector.dot(&target_arr);
        let prob_positive = sigmoid(dot_product);
        let grad_positive = prob_positive - 1.0;

        for (context_idx, &context_id) in context_ids.iter().enumerate() {
            let _context_embedding = &context_embeddings[context_idx];
            for i in 0..self.config.embedding_dim {
                let mut grad = learning_rate * grad_positive * target_embedding[i];
                if let Some(l2_reg) = self.config.l2_regularization {
                    grad += learning_rate * l2_reg as f32 * self.embeddings[[context_id, i]];
                }
                self.accumulate_gradient(accum, context_id, grad, i);
            }
        }

        for i in 0..self.config.embedding_dim {
            let mut grad = learning_rate * grad_positive * context_vector[i];
            if let Some(l2_reg) = self.config.l2_regularization {
                grad += learning_rate * l2_reg as f32 * target_embedding[i];
            }
            self.accumulate_gradient(accum, target_id, grad, i);
        }

        for &neg_id in negative_samples {
            let neg_embedding: Vec<f32> = self.embeddings.row(neg_id).to_vec();
            let neg_arr = Array::from_shape_vec((self.config.embedding_dim,), neg_embedding.clone()).unwrap();
            let dot_product_neg = context_vector.dot(&neg_arr);
            let prob_negative = sigmoid(dot_product_neg);
            let grad_negative = prob_negative;

            for &context_id in context_ids {
                for i in 0..self.config.embedding_dim {
                    let mut grad = learning_rate * grad_negative * neg_embedding[i];
                    if let Some(l2_reg) = self.config.l2_regularization {
                        grad += learning_rate * l2_reg as f32 * self.embeddings[[context_id, i]];
                    }
                    self.accumulate_gradient(accum, context_id, grad, i);
                }
            }

            for i in 0..self.config.embedding_dim {
                let mut grad = learning_rate * grad_negative * context_vector[i];
                if let Some(l2_reg) = self.config.l2_regularization {
                    grad += learning_rate * l2_reg as f32 * neg_embedding[i];
                }
                self.accumulate_gradient(accum, neg_id, grad, i);
            }
        }

        let mut loss = -prob_positive.ln();
        for &neg_id in negative_samples {
            let neg_embedding: Vec<f32> = self.embeddings.row(neg_id).to_vec();
            let neg_arr = Array::from_shape_vec((self.config.embedding_dim,), neg_embedding.clone()).unwrap();
            let dot_product_neg = context_vector.dot(&neg_arr);
            loss += -sigmoid(-dot_product_neg).ln();
        }
        loss
    }

    pub fn get_embedding(&self, word: &str, data: &TrainingData) -> Option<Array1<f32>> {
        if let Some(&word_id) = data.vocab.get(word) {
            Some(self.embeddings.row(word_id).to_owned())
        } else {
            None
        }
    }

    pub fn save_embeddings(&self, path: &str, data: &TrainingData) -> Result<(), String> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        
        for (word_id, word) in data.reverse_vocab.iter().enumerate() {
            let embedding = self.embeddings.row(word_id);
            let embedding_str = embedding.iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            
            writeln!(file, "{}\t{}", word, embedding_str).map_err(|e| e.to_string())?;
        }
        
        Ok(())
    }

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
    
    pub fn evaluate(&self, data: &TrainingData, validation_data: &ValidationData) -> EvaluationMetrics {
        let mut correct_pairs = 0;
        let mut total_pairs = 0;
        let mut total_similarity = 0.0;
        let mut similarities = Vec::new();
        
        // Evaluate word similarity
        for (word1, word2) in validation_data.positive_pairs.iter() {
            if let Some(sim) = self.similarity(word1, word2, data) {
                similarities.push(sim);
                total_similarity += sim;
                correct_pairs += 1;
            }
            total_pairs += 1;
        }
        
        for (word1, word2) in validation_data.negative_pairs.iter() {
            if let Some(sim) = self.similarity(word1, word2, data) {
                similarities.push(sim);
                total_similarity += sim;
                correct_pairs += 1;
            }
            total_pairs += 1;
        }
        
        let accuracy = if total_pairs > 0 { correct_pairs as f32 / total_pairs as f32 } else { 0.0 };
        let mean_similarity = if !similarities.is_empty() { total_similarity / similarities.len() as f32 } else { 0.0 };
        
        // Calculate embedding quality score based on various metrics
        let embedding_quality_score = self.calculate_embedding_quality(data);
        
        EvaluationMetrics {
            accuracy,
            precision: accuracy,  // Simplified for now
            recall: accuracy,     // Simplified for now
            f1_score: accuracy,  // Simplified for now
            mean_similarity,
            embedding_quality_score,
        }
    }
    
    fn calculate_embedding_quality(&self, data: &TrainingData) -> f32 {
        let mut total_norm = 0.0;
        let mut count = 0;
        let mut total_variance = 0.0;
        
        for (word_id, _) in data.reverse_vocab.iter().enumerate() {
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

    pub fn normalize_embeddings(&mut self) {
        for mut row in self.embeddings.rows_mut() {
            let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.map_inplace(|x| *x /= norm);
            }
        }
    }

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
                
                if s1.len() >= 1 && s2.len() >= 1 && s3.len() >= 1 && s4.len() >= 1 {
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
}

pub fn build_vocab(sentences: &[Vec<String>]) -> (HashMap<String, usize>, Vec<String>) {
    let mut vocab = HashMap::new();
    let mut reverse_vocab = Vec::new();
    let mut vocab_counter = 0;
    
    for sentence in sentences {
        for word in sentence {
            if !vocab.contains_key(word) {
                vocab.insert(word.clone(), vocab_counter);
                reverse_vocab.push(word.clone());
                vocab_counter += 1;
            }
        }
    }
    
    (vocab, reverse_vocab)
}

// Helper function for sigmoid activation
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Negative sampling utility
fn get_negative_samples(vocab_size: usize, num_samples: usize, target_id: usize, rng: &mut rand::rngs::ThreadRng) -> Vec<usize> {
    let mut samples = Vec::new();
    let dist = Uniform::new(0, vocab_size);
    
    while samples.len() < num_samples {
        let candidate = rng.sample(dist);
        if candidate != target_id && !samples.contains(&candidate) {
            samples.push(candidate);
        }
    }
    
    samples
}

// Learning rate scheduling
impl EmbeddingModel {
    fn get_learning_rate(&self, epoch: usize, total_epochs: usize) -> f32 {
        match self.config.lr_schedule {
            LearningRateSchedule::Constant => self.config.learning_rate as f32,
            LearningRateSchedule::Exponential { decay_rate } => {
                (self.config.learning_rate * decay_rate.powi(epoch as i32)) as f32
            }
            LearningRateSchedule::Step { step_size, gamma } => {
                let num_steps = epoch / step_size;
                (self.config.learning_rate * gamma.powi(num_steps as i32)) as f32
            }
            LearningRateSchedule::Cosine { t_max } => {
                let t = epoch as f32 / std::cmp::min(t_max, total_epochs) as f32;
                let lr = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                self.config.learning_rate as f32 * lr
            }
        }
    }
    
    fn should_early_stop(&self, _epoch: usize, current_loss: f32, best_loss: f32, patience_counter: &mut usize) -> bool {
        if let Some(config) = &self.config.early_stopping {
            if current_loss < best_loss - config.min_delta as f32 {
                *patience_counter = 0;
                false
            } else {
                *patience_counter += 1;
                *patience_counter >= config.patience
            }
        } else {
            false
        }
    }
    
    fn clip_gradient(&self, grad: f32) -> f32 {
        if let Some(max_norm) = self.config.gradient_clip {
            grad.clamp(-max_norm, max_norm)
        } else {
            grad
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub mean_similarity: f32,
    pub embedding_quality_score: f32,
}

#[derive(Debug, Clone)]
pub struct ValidationData {
    pub positive_pairs: Vec<(String, String)>,
    pub negative_pairs: Vec<(String, String)>,
    pub analogies: Vec<(String, String, String, String)>,  // (word1, word2, word3, word4) for word1 - word2 + word3 = word4
}

#[derive(Debug, Clone)]
pub struct TextProcessor {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub remove_numbers: bool,
    pub remove_stop_words: bool,
    pub remove_html: bool,
    pub remove_urls: bool,
    pub expand_contractions: bool,
    pub language: String,
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            remove_numbers: false,
            remove_stop_words: false,
            remove_html: false,
            remove_urls: false,
            expand_contractions: false,
            language: "en".to_string(),
        }
    }
}

impl TextProcessor {
    pub fn process_text(&self, text: &str) -> Vec<Vec<String>> {
        let mut text = text.to_string();

        // Remove HTML tags
        if self.remove_html {
            text = Self::strip_html(&text);
        }

        // Remove URLs
        if self.remove_urls {
            text = Self::strip_urls(&text);
        }

        let mut sentences = Vec::new();

        // Split into sentences
        for sentence in text.split(['.', '!', '?', '\n']) {
            if !sentence.trim().is_empty() {
                let mut processed_words = Vec::new();

                // Split into words and process each word
                for word in sentence.split_whitespace() {
                    let processed_word = self.process_word(word);
                    if !processed_word.is_empty() {
                        for subword in processed_word.split_whitespace() {
                            processed_words.push(subword.to_string());
                        }
                    }
                }

                if !processed_words.is_empty() {
                    sentences.push(processed_words);
                }
            }
        }

        sentences
    }

    fn strip_html(text: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;
        for c in text.chars() {
            if c == '<' {
                in_tag = true;
            } else if c == '>' {
                in_tag = false;
            } else if !in_tag {
                result.push(c);
            }
        }
        result
    }

    fn strip_urls(text: &str) -> String {
        text.split_whitespace()
            .filter(|word| !(word.starts_with("http://") || word.starts_with("https://") || word.starts_with("www.")))
            .collect::<Vec<&str>>()
            .join(" ")
    }

    fn process_word(&self, word: &str) -> String {
        let mut result = word.to_string();

        // Expand contractions
        if self.expand_contractions {
            result = Self::expand_contraction(&result);
        }

        // Convert to lowercase
        if self.lowercase {
            result = result.to_lowercase();
        }

        // Remove punctuation
        if self.remove_punctuation {
            result = result.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect::<String>()
                .trim()
                .to_string();
        }

        // Remove numbers
        if self.remove_numbers {
            result = result.chars()
                .filter(|c| !c.is_ascii_digit())
                .collect::<String>();
        }

        // Remove empty strings
        if result.is_empty() {
            return String::new();
        }

        result
    }

    fn expand_contraction(word: &str) -> String {
        match word.to_lowercase().as_str() {
            "can't" => "cannot".to_string(),
            "won't" => "will not".to_string(),
            "n't" => " not".to_string(),
            "'re" => " are".to_string(),
            "'ve" => " have".to_string(),
            "'ll" => " will".to_string(),
            "'d" => " would".to_string(),
            "'m" => " am".to_string(),
            "i'm" => "i am".to_string(),
            "don't" => "do not".to_string(),
            "doesn't" => "does not".to_string(),
            "didn't" => "did not".to_string(),
            "isn't" => "is not".to_string(),
            "aren't" => "are not".to_string(),
            "wasn't" => "was not".to_string(),
            "weren't" => "were not".to_string(),
            "haven't" => "have not".to_string(),
            "hasn't" => "has not".to_string(),
            "hadn't" => "had not".to_string(),
            "wouldn't" => "would not".to_string(),
            "couldn't" => "could not".to_string(),
            "shouldn't" => "should not".to_string(),
            "let's" => "let us".to_string(),
            "that's" => "that is".to_string(),
            "who's" => "who is".to_string(),
            "what's" => "what is".to_string(),
            "here's" => "here is".to_string(),
            "there's" => "there is".to_string(),
            "where's" => "where is".to_string(),
            "it's" => "it is".to_string(),
            _ => word.to_string(),
        }
    }
    
    pub fn detect_language(&self, text: &str) -> String {
        // Simple heuristic for language detection
        // This is a very basic implementation - in practice, you'd use more sophisticated methods
        
        let english_stop_words = ["the", "and", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"];
        
        let words_vec: Vec<&str> = text.split_whitespace().collect();
        let words = &words_vec;
        let mut english_count = 0;
        
        for word in words {
            let lower_word = word.to_lowercase();
            if english_stop_words.contains(&lower_word.as_str()) {
                english_count += 1;
            }
        }
        
        // If more than 20% of words are common English stop words, assume English
        if english_count > words.len() / 5 {
            "en".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

pub fn load_text_data(text: &str) -> Vec<Vec<String>> {
    let processor = TextProcessor::default();
    processor.process_text(text)
}

pub fn load_text_data_advanced(text: &str, processor: &TextProcessor) -> Vec<Vec<String>> {
    processor.process_text(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_vocab() {
        let sentences = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["hello".to_string(), "rust".to_string()],
        ];
        
        let (vocab, reverse_vocab) = build_vocab(&sentences);
        
        assert_eq!(vocab.len(), 3);
        assert_eq!(reverse_vocab.len(), 3);
        assert_eq!(vocab.get("hello"), Some(&0));
        assert_eq!(vocab.get("world"), Some(&1));
        assert_eq!(vocab.get("rust"), Some(&2));
    }

    #[test]
    fn test_load_text_data() {
        let text = "Hello world! This is a test.";
        let sentences = load_text_data(text);

        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], vec!["hello", "world"]);
        assert_eq!(sentences[1], vec!["this", "is", "a", "test"]);
    }

    fn make_test_data() -> TrainingData {
        let text = "the cat sat on the mat. the dog sat on the log. the cat chased the dog.";
        let sentences = load_text_data(text);
        let (vocab, reverse_vocab) = build_vocab(&sentences);
        TrainingData { sentences, vocab, reverse_vocab }
    }

    fn test_config(model_type: ModelType) -> TrainingConfig {
        TrainingConfig {
            embedding_dim: 8,
            learning_rate: 0.1,
            epochs: 2,
            batch_size: 4,
            context_window: 1,
            negative_samples: 2,
            model_type,
            lr_schedule: LearningRateSchedule::Constant,
            early_stopping: None,
            l2_regularization: None,
            dropout_rate: None,
            gradient_clip: None,
        }
    }

    #[test]
    fn test_train_skipgram() {
        let data = make_test_data();
        let config = test_config(ModelType::SkipGram);
        let mut model = EmbeddingModel::new(config, data.vocab.len());

        assert!(model.train(&data).is_ok());

        // Embeddings should exist for known words
        assert!(model.get_embedding("cat", &data).is_some());
        assert!(model.get_embedding("dog", &data).is_some());
        assert!(model.get_embedding("the", &data).is_some());

        // Similarity should return a value for known pairs
        assert!(model.similarity("cat", "dog", &data).is_some());
    }

    #[test]
    fn test_train_cbow() {
        let data = make_test_data();
        let config = test_config(ModelType::Cbow);
        let mut model = EmbeddingModel::new(config, data.vocab.len());

        assert!(model.train(&data).is_ok());

        assert!(model.get_embedding("cat", &data).is_some());
        assert!(model.get_embedding("dog", &data).is_some());
        assert!(model.similarity("cat", "dog", &data).is_some());
    }

    #[test]
    fn test_save_embeddings() {
        let data = make_test_data();
        let config = test_config(ModelType::SkipGram);
        let mut model = EmbeddingModel::new(config, data.vocab.len());
        model.train(&data).unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_embeddings_save.txt");
        let path_str = path.to_str().unwrap();

        assert!(model.save_embeddings(path_str, &data).is_ok());
        let contents = std::fs::read_to_string(path_str).unwrap();
        assert!(contents.contains("cat"));
        assert!(contents.contains("dog"));

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_similarity_unknown_word() {
        let data = make_test_data();
        let config = test_config(ModelType::SkipGram);
        let mut model = EmbeddingModel::new(config, data.vocab.len());
        model.train(&data).unwrap();

        assert!(model.similarity("cat", "nonexistent", &data).is_none());
        assert!(model.similarity("nonexistent", "dog", &data).is_none());
    }

    #[test]
    fn test_strip_html() {
        let processor = TextProcessor {
            remove_html: true,
            remove_punctuation: false,
            lowercase: false,
            ..TextProcessor::default()
        };
        let text = "<p>Hello world!</p> This is a <b>test</b>.";
        let sentences = processor.process_text(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], vec!["Hello", "world"]);
        assert_eq!(sentences[1], vec!["This", "is", "a", "test"]);
    }

    #[test]
    fn test_strip_urls() {
        let processor = TextProcessor {
            remove_urls: true,
            remove_punctuation: true,
            lowercase: true,
            ..TextProcessor::default()
        };
        let text = "Visit https://example.com for info. See www.test.org too.";
        let sentences = processor.process_text(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], vec!["visit", "for", "info"]);
        assert_eq!(sentences[1], vec!["see", "too"]);
    }

    #[test]
    fn test_expand_contractions() {
        let processor = TextProcessor {
            expand_contractions: true,
            remove_punctuation: true,
            lowercase: true,
            ..TextProcessor::default()
        };
        let text = "I can't do this. It's a test.";
        let sentences = processor.process_text(text);
        assert_eq!(sentences.len(), 2);
        // "can't" -> "cannot", then punctuation stripped
        assert_eq!(sentences[0], vec!["i", "cannot", "do", "this"]);
        assert_eq!(sentences[1], vec!["it", "is", "a", "test"]);
    }

    #[test]
    fn test_normalize_embeddings() {
        let data = make_test_data();
        let config = test_config(ModelType::SkipGram);
        let mut model = EmbeddingModel::new(config, data.vocab.len());
        model.train(&data).unwrap();
        model.normalize_embeddings();

        for row in model.embeddings.rows() {
            let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
        }
    }

    #[test]
    fn test_analogy_unknown_word() {
        let data = make_test_data();
        let config = test_config(ModelType::SkipGram);
        let mut model = EmbeddingModel::new(config, data.vocab.len());
        model.train(&data).unwrap();

        assert!(model.analogy("unknown", "cat", "dog", &data, 1).is_empty());
    }

    #[test]
    fn test_split_data() {
        let sentences = vec![
            vec!["a".to_string()],
            vec!["b".to_string()],
            vec!["c".to_string()],
            vec!["d".to_string()],
            vec!["e".to_string()],
            vec!["f".to_string()],
            vec!["g".to_string()],
            vec!["h".to_string()],
            vec!["i".to_string()],
            vec!["j".to_string()],
        ];
        let config = test_config(ModelType::SkipGram);
        let model = EmbeddingModel::new(config, 1);
        let (train, val) = model.split_data(&sentences, 0.7);
        assert_eq!(train.len(), 7);
        assert_eq!(val.len(), 3);
    }

    #[test]
    fn test_gradient_clipping() {
        let data = make_test_data();
        let mut config = test_config(ModelType::SkipGram);
        config.gradient_clip = Some(0.001);
        let mut model = EmbeddingModel::new(config, data.vocab.len());

        // Training should still succeed with aggressive clipping
        assert!(model.train(&data).is_ok());
        assert!(model.get_embedding("cat", &data).is_some());
    }

    #[test]
    fn test_mini_batch_processing() {
        let data = make_test_data();
        // Test with batch_size = 1 (equivalent to old behavior)
        let mut config1 = test_config(ModelType::SkipGram);
        config1.batch_size = 1;
        let mut model1 = EmbeddingModel::new(config1, data.vocab.len());
        assert!(model1.train(&data).is_ok());

        // Test with batch_size = 8 (actual mini-batch)
        let mut config8 = test_config(ModelType::SkipGram);
        config8.batch_size = 8;
        let mut model8 = EmbeddingModel::new(config8, data.vocab.len());
        assert!(model8.train(&data).is_ok());

        // Both should produce embeddings for known words
        assert!(model1.get_embedding("cat", &data).is_some());
        assert!(model8.get_embedding("cat", &data).is_some());
    }
}