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
    use serde_json::Value;

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
        let embeddings = Array::zeros((vocab_size, config.embedding_dim));
        
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
            
            let total_loss = 0.0;
            let mut num_updates = 0;
            
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
            // Use current learning rate
            let temp_lr = self.config.learning_rate as f32;
            let current_lr_f32 = current_lr;
            self.update_skipgram_with_lr(target_id, context_id, &negative_samples, current_lr_f32);
            num_updates += 1;
                                }
                            }
                        }
                    }
                }
            }
            
            let avg_loss = total_loss / num_updates as f32;
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
            
            let total_loss = 0.0;
            let mut num_updates = 0;
            
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
                            
                            // Use current learning rate
                            let current_lr_f32 = current_lr;
                            self.update_cbow_with_lr(target_id, &context_ids, &negative_samples, current_lr_f32);
                            num_updates += 1;
                        }
                    }
                }
            }
            
            let avg_loss = total_loss / num_updates as f32;
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

    fn update_skipgram(&mut self, target_id: usize, context_id: usize, negative_samples: &[usize]) {
        self.update_skipgram_with_lr(target_id, context_id, negative_samples, self.config.learning_rate as f32);
    }
    
    fn update_skipgram_with_lr(&mut self, target_id: usize, context_id: usize, negative_samples: &[usize], learning_rate: f32) {
        // Clone embeddings to avoid borrow checker issues
        let target_embedding: Vec<f32> = self.embeddings.row(target_id).to_vec();
        let context_embedding: Vec<f32> = self.embeddings.row(context_id).to_vec();
        
        // Positive sample loss
        let dot_product: f32 = target_embedding.iter().zip(context_embedding.iter()).map(|(&a, &b)| a * b).sum();
        let prob_positive = sigmoid(dot_product);
        
        // Update positive sample
        let grad_positive = prob_positive - 1.0;
        for i in 0..self.config.embedding_dim {
            let mut grad = learning_rate * grad_positive * context_embedding[i];
            
            // Apply L2 regularization
            if let Some(l2_reg) = self.config.l2_regularization {
                grad += learning_rate * l2_reg as f32 * target_embedding[i];
            }
            
            self.embeddings[[target_id, i]] -= grad;
            
            let mut grad_context = learning_rate * grad_positive * target_embedding[i];
            if let Some(l2_reg) = self.config.l2_regularization {
                grad_context += learning_rate * l2_reg as f32 * context_embedding[i];
            }
            
            self.embeddings[[context_id, i]] -= grad_context;
        }
        
        // Negative samples
        for &neg_id in negative_samples {
            let neg_embedding: Vec<f32> = self.embeddings.row(neg_id).to_vec();
            let dot_product_neg: f32 = target_embedding.iter().zip(neg_embedding.iter()).map(|(&a, &b)| a * b).sum();
            let prob_negative = sigmoid(dot_product_neg);
            
            // Update negative sample
            let grad_negative = prob_negative;
            for i in 0..self.config.embedding_dim {
                let mut grad = learning_rate * grad_negative * neg_embedding[i];
                if let Some(l2_reg) = self.config.l2_regularization {
                    grad += learning_rate * l2_reg as f32 * target_embedding[i];
                }
                
                self.embeddings[[target_id, i]] -= grad;
                
                let mut grad_neg = learning_rate * grad_negative * target_embedding[i];
                if let Some(l2_reg) = self.config.l2_regularization {
                    grad_neg += learning_rate * l2_reg as f32 * neg_embedding[i];
                }
                
                self.embeddings[[neg_id, i]] -= grad_neg;
            }
        }
        
        // Apply dropout if enabled
        if let Some(dropout_rate) = self.config.dropout_rate {
            self.apply_dropout(target_id, dropout_rate);
            self.apply_dropout(context_id, dropout_rate);
            for &neg_id in negative_samples {
                self.apply_dropout(neg_id, dropout_rate);
            }
        }
    }

    fn update_cbow(&mut self, target_id: usize, context_ids: &[usize], negative_samples: &[usize]) {
        self.update_cbow_with_lr(target_id, context_ids, negative_samples, self.config.learning_rate as f32);
    }
    
    fn update_cbow_with_lr(&mut self, target_id: usize, context_ids: &[usize], negative_samples: &[usize], learning_rate: f32) {
        // Calculate context vector (mean pooling)
        let mut context_vector = Array::zeros(self.config.embedding_dim);
        let mut context_embeddings: Vec<Vec<f32>> = Vec::new();
        
        for &context_id in context_ids {
            let context_embedding: Vec<f32> = self.embeddings.row(context_id).to_vec();
            context_embeddings.push(context_embedding.clone());
            let context_arr = Array::from_shape_vec((self.config.embedding_dim,), context_embedding).unwrap();
            context_vector = context_vector + &context_arr;
        }
        context_vector = context_vector / context_ids.len() as f32;
        
        // Positive sample
        let target_embedding: Vec<f32> = self.embeddings.row(target_id).to_vec();
        let target_arr = Array::from_shape_vec((self.config.embedding_dim,), target_embedding.clone()).unwrap();
        let dot_product = context_vector.dot(&target_arr);
        let prob_positive = sigmoid(dot_product);
        let grad_positive = prob_positive - 1.0;
        
        // Update positive sample
        for (context_idx, &context_id) in context_ids.iter().enumerate() {
            let _context_embedding = &context_embeddings[context_idx];
            for i in 0..self.config.embedding_dim {
                let mut grad = learning_rate * grad_positive * target_embedding[i];
                if let Some(l2_reg) = self.config.l2_regularization {
                    grad += learning_rate * l2_reg as f32 * self.embeddings[[context_id, i]];
                }
                self.embeddings[[context_id, i]] -= grad;
            }
        }
        
        for i in 0..self.config.embedding_dim {
            let mut grad = learning_rate * grad_positive * context_vector[i];
            if let Some(l2_reg) = self.config.l2_regularization {
                grad += learning_rate * l2_reg as f32 * target_embedding[i];
            }
            self.embeddings[[target_id, i]] -= grad;
        }
        
        // Negative samples
        for &neg_id in negative_samples {
            let neg_embedding: Vec<f32> = self.embeddings.row(neg_id).to_vec();
            let neg_arr = Array::from_shape_vec((self.config.embedding_dim,), neg_embedding.clone()).unwrap();
            let dot_product_neg = context_vector.dot(&neg_arr);
            let prob_negative = sigmoid(dot_product_neg);
            let grad_negative = prob_negative;
            
            // Update negative sample
            for &context_id in context_ids {
                for i in 0..self.config.embedding_dim {
                    let mut grad = learning_rate * grad_negative * neg_embedding[i];
                    if let Some(l2_reg) = self.config.l2_regularization {
                        grad += learning_rate * l2_reg as f32 * self.embeddings[[context_id, i]];
                    }
                    self.embeddings[[context_id, i]] -= grad;
                }
            }
            
            for i in 0..self.config.embedding_dim {
                let mut grad = learning_rate * grad_negative * context_vector[i];
                if let Some(l2_reg) = self.config.l2_regularization {
                    grad += learning_rate * l2_reg as f32 * neg_embedding[i];
                }
                self.embeddings[[neg_id, i]] -= grad;
            }
        }
        
        // Apply dropout if enabled
        if let Some(dropout_rate) = self.config.dropout_rate {
            for &context_id in context_ids {
                self.apply_dropout(context_id, dropout_rate);
            }
            self.apply_dropout(target_id, dropout_rate);
            for &neg_id in negative_samples {
                self.apply_dropout(neg_id, dropout_rate);
            }
        }
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
    
    fn apply_dropout(&mut self, word_id: usize, dropout_rate: f32) {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0.0, 1.0);
        
        for i in 0..self.config.embedding_dim {
            if rng.sample(dist) < dropout_rate {
                self.embeddings[[word_id, i]] = 0.0;
            }
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
    pub language: String,
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            remove_numbers: false,
            remove_stop_words: false,
            language: "en".to_string(),
        }
    }
}

impl TextProcessor {
    pub fn process_text(&self, text: &str) -> Vec<Vec<String>> {
        let mut sentences = Vec::new();
        
        // Split into sentences
        for sentence in text.split(['.', '!', '?', '\n']) {
            if !sentence.trim().is_empty() {
                let mut processed_words = Vec::new();
                
                // Split into words and process each word
                for word in sentence.split_whitespace() {
                    let processed_word = self.process_word(word);
                    if !processed_word.is_empty() {
                        processed_words.push(processed_word);
                    }
                }
                
                if !processed_words.is_empty() {
                    sentences.push(processed_words);
                }
            }
        }
        
        sentences
    }
    
    fn process_word(&self, word: &str) -> String {
        let mut result = word.to_string();
        
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
}