use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use std::collections::HashMap;
use tracing::info;
use crate::{EmbeddingModel, TrainingData};

impl EmbeddingModel {
    pub(crate) fn train_skipgram(&mut self, data: &TrainingData) -> Result<(), String> {
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

            for sentence in data.sentences.iter() {
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

                        for (context_idx, context_word) in sentence.iter().enumerate().skip(start).take(end - start) {
                            if context_idx != target_idx
                                && let Some(&context_id) = data.vocab.get(context_word) {
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

            // Apply any remaining gradients
            if batch_count > 0 {
                self.apply_gradient_batch(&mut accum, batch_count);
            }

            let avg_loss = total_loss / num_updates.max(1) as f32;
            info!("Epoch {} completed. Average loss: {:.4}", epoch + 1, avg_loss);
            self.training_history.record_epoch(epoch + 1, avg_loss, current_lr as f64);

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

    pub(crate) fn train_cbow(&mut self, data: &TrainingData) -> Result<(), String> {
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

            for sentence in data.sentences.iter() {
                for (target_idx, target_word) in sentence.iter().enumerate() {
                    if let Some(&target_id) = data.vocab.get(target_word) {
                        let start = target_idx.saturating_sub(self.config.context_window);
                        let end = std::cmp::min(target_idx + self.config.context_window + 1, sentence.len());

                        let mut context_ids = Vec::new();
                        for (context_idx, context_word) in sentence.iter().enumerate().skip(start).take(end - start) {
                            if context_idx != target_idx
                                && let Some(&context_id) = data.vocab.get(context_word) {
                                    context_ids.push(context_id);
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
            self.training_history.record_epoch(epoch + 1, avg_loss, current_lr as f64);

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

        for (i, (&t, &c)) in target_embedding.iter().zip(context_embedding.iter()).enumerate() {
            let grad = self.l2_grad(learning_rate * grad_positive * c, learning_rate, t);
            self.accumulate_gradient(accum, target_id, grad, i);

            let grad_context = self.l2_grad(learning_rate * grad_positive * t, learning_rate, c);
            self.accumulate_gradient(accum, context_id, grad_context, i);
        }

        let mut loss = -prob_positive.ln();
        for &neg_id in negative_samples {
            let neg_embedding: Vec<f32> = self.embeddings.row(neg_id).to_vec();
            let dot_product_neg: f32 = target_embedding.iter().zip(neg_embedding.iter()).map(|(&a, &b)| a * b).sum();
            let prob_negative = sigmoid(dot_product_neg);
            let grad_negative = prob_negative;
            loss += -(1.0 - prob_negative).ln();

            for (i, (&t, &n)) in target_embedding.iter().zip(neg_embedding.iter()).enumerate() {
                let grad = self.l2_grad(learning_rate * grad_negative * n, learning_rate, t);
                self.accumulate_gradient(accum, target_id, grad, i);

                let grad_neg = self.l2_grad(learning_rate * grad_negative * t, learning_rate, n);
                self.accumulate_gradient(accum, neg_id, grad_neg, i);
            }
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
            context_vector += &context_arr;
        }
        context_vector /= context_ids.len() as f32;

        let target_embedding: Vec<f32> = self.embeddings.row(target_id).to_vec();
        let target_arr = Array::from_shape_vec((self.config.embedding_dim,), target_embedding.clone()).unwrap();
        let dot_product = context_vector.dot(&target_arr);
        let prob_positive = sigmoid(dot_product);
        let grad_positive = prob_positive - 1.0;

        for (context_idx, &context_id) in context_ids.iter().enumerate() {
            let _context_embedding = &context_embeddings[context_idx];
            for (i, &target_val) in target_embedding.iter().enumerate() {
                let weight = self.embeddings[[context_id, i]];
                let grad = self.l2_grad(learning_rate * grad_positive * target_val, learning_rate, weight);
                self.accumulate_gradient(accum, context_id, grad, i);
            }
        }

        for (i, &context_val) in context_vector.iter().enumerate() {
            let grad = self.l2_grad(learning_rate * grad_positive * context_val, learning_rate, target_embedding[i]);
            self.accumulate_gradient(accum, target_id, grad, i);
        }

        let mut loss = -prob_positive.ln();
        for &neg_id in negative_samples {
            let neg_embedding: Vec<f32> = self.embeddings.row(neg_id).to_vec();
            let neg_arr = Array::from_shape_vec((self.config.embedding_dim,), neg_embedding.clone()).unwrap();
            let dot_product_neg = context_vector.dot(&neg_arr);
            let prob_negative = sigmoid(dot_product_neg);
            let grad_negative = prob_negative;
            loss += -(1.0 - prob_negative).ln();

            for &context_id in context_ids {
                for (i, &neg_val) in neg_embedding.iter().enumerate() {
                    let weight = self.embeddings[[context_id, i]];
                    let grad = self.l2_grad(learning_rate * grad_negative * neg_val, learning_rate, weight);
                    self.accumulate_gradient(accum, context_id, grad, i);
                }
            }

            for (i, &context_val) in context_vector.iter().enumerate() {
                let grad = self.l2_grad(learning_rate * grad_negative * context_val, learning_rate, neg_embedding[i]);
                self.accumulate_gradient(accum, neg_id, grad, i);
            }
        }
        loss
    }

    fn get_learning_rate(&self, epoch: usize, total_epochs: usize) -> f32 {
        match self.config.lr_schedule {
            crate::config::LearningRateSchedule::Constant => self.config.learning_rate as f32,
            crate::config::LearningRateSchedule::Exponential { decay_rate } => {
                (self.config.learning_rate * decay_rate.powi(epoch as i32)) as f32
            }
            crate::config::LearningRateSchedule::Step { step_size, gamma } => {
                let num_steps = epoch / step_size;
                (self.config.learning_rate * gamma.powi(num_steps as i32)) as f32
            }
            crate::config::LearningRateSchedule::Cosine { t_max } => {
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
    
    fn l2_grad(&self, grad: f32, learning_rate: f32, weight: f32) -> f32 {
        if let Some(l2_reg) = self.config.l2_regularization {
            grad + learning_rate * l2_reg as f32 * weight
        } else {
            grad
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

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

/// Supports real-time incremental training by updating a model with new
/// sentences as they arrive, without requiring a full retrain.
pub struct IncrementalTrainer;

impl IncrementalTrainer {
    /// Updates `model` by training on a batch of new sentences for a small
    /// number of epochs. New vocabulary words are added automatically.
    ///
    /// Returns the updated `TrainingData` reflecting any new vocabulary.
    pub fn update(
        model: &mut EmbeddingModel,
        data: &mut TrainingData,
        new_sentences: &[Vec<String>],
        mini_epochs: usize,
    ) -> Result<(), String> {
        // Update vocabulary with any new words
        let new_words: Vec<String> = new_sentences
            .iter()
            .flat_map(|s| s.iter().cloned())
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();
        model.incremental_vocab_update(&new_words, data)?;

        // Append new sentences to existing data
        data.sentences.extend(new_sentences.iter().cloned());

        // Temporarily reduce epochs for quick incremental update
        let original_epochs = model.config.epochs;
        model.config.epochs = mini_epochs;
        model.train(data)?;
        model.config.epochs = original_epochs;

        Ok(())
    }

    /// Streams sentences from an iterator and trains the model incrementally
    /// in micro-batches. Useful for real-time data feeds.
    pub fn stream_train<I>(
        model: &mut EmbeddingModel,
        data: &mut TrainingData,
        sentences: I,
        batch_size: usize,
        mini_epochs: usize,
    ) -> Result<(), String>
    where
        I: Iterator<Item = Vec<String>>,
    {
        let mut batch = Vec::with_capacity(batch_size);
        for sentence in sentences {
            batch.push(sentence);
            if batch.len() >= batch_size {
                Self::update(model, data, &batch, mini_epochs)?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            Self::update(model, data, &batch, mini_epochs)?;
        }
        Ok(())
    }
}
