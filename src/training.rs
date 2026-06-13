use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::distributions::WeightedIndex;
use rayon::prelude::*;
use std::collections::HashMap;
use tracing::info;
use crate::{EmbeddingModel, TrainingData};

impl EmbeddingModel {
    pub(crate) fn train_skipgram(&mut self, data: &TrainingData) -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut best_loss = f32::MAX;
        let mut patience_counter = 0;

        // Pre-compute negative-sampling distribution if enabled
        let neg_dist = if self.config.use_unigram_negative_sampling && !data.word_freq.is_empty() {
            Some(build_negative_sampling_dist(&data.word_freq))
        } else {
            None
        };
        let neg_sampler = neg_dist.as_ref().and_then(|d| WeightedIndex::new(d).ok());

        // Pre-compute sub-sampling state
        let total_words = data.total_word_count() as f64;
        let subsample_threshold = self.config.subsample_threshold;

        for epoch in 0..self.config.epochs {
            info!("Epoch {}/{}", epoch + 1, self.config.epochs);

            let current_lr = self.get_learning_rate(epoch, self.config.epochs);
            info!("Current learning rate: {:.6}", current_lr);

            let (total_loss, num_updates) = if self.config.use_parallel {
                self.epoch_skipgram_parallel(data, neg_sampler.as_ref(), total_words, subsample_threshold, current_lr)
            } else {
                let mut total_loss = 0.0;
                let mut num_updates = 0;
                let mut batch_count = 0;
                let mut batch_loss = 0.0;
                let mut num_batches = 0;
                let mut accum = self.new_gradient_accumulator();

                for sentence in data.sentences.iter() {
                    for (target_idx, target_word) in sentence.iter().enumerate() {
                        if let Some(&target_id) = data.vocab.get(target_word) {
                            // Sub-sampling: skip very frequent words
                            if should_subsample(target_id, &data.word_freq, subsample_threshold, total_words, &mut rng) {
                                continue;
                            }

                            let start = target_idx.saturating_sub(self.config.context_window);
                            let end = std::cmp::min(target_idx + self.config.context_window + 1, sentence.len());

                            // Get negative samples
                            let negative_samples = get_negative_samples(
                                self.vocab_size,
                                self.config.negative_samples,
                                target_id,
                                neg_sampler.as_ref(),
                                &mut rng
                            );

                            for (context_idx, context_word) in sentence.iter().enumerate().skip(start).take(end - start) {
                                if context_idx != target_idx
                                    && let Some(&context_id) = data.vocab.get(context_word) {
                                        // Sub-sampling for context words too
                                        if should_subsample(context_id, &data.word_freq, subsample_threshold, total_words, &mut rng) {
                                            continue;
                                        }

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

                (total_loss, num_updates)
            };

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

            // Save checkpoint if configured
            if let Some(interval) = self.config.checkpoint_interval {
                if (epoch + 1) % interval == 0 {
                    let dir = self.config.checkpoint_path.as_deref().unwrap_or(".");
                    let path = format!("{}/checkpoint_epoch_{}.json", dir, epoch + 1);
                    if let Err(e) = self.save_checkpoint(&path, epoch + 1, best_loss) {
                        tracing::warn!("Failed to save checkpoint: {}", e);
                    } else {
                        tracing::info!("Checkpoint saved: {}", path);
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn train_cbow(&mut self, data: &TrainingData) -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut best_loss = f32::MAX;
        let mut patience_counter = 0;

        // Pre-compute negative-sampling distribution if enabled
        let neg_dist = if self.config.use_unigram_negative_sampling && !data.word_freq.is_empty() {
            Some(build_negative_sampling_dist(&data.word_freq))
        } else {
            None
        };
        let neg_sampler = neg_dist.as_ref().and_then(|d| WeightedIndex::new(d).ok());

        // Pre-compute sub-sampling state
        let total_words = data.total_word_count() as f64;
        let subsample_threshold = self.config.subsample_threshold;

        for epoch in 0..self.config.epochs {
            info!("Epoch {}/{}", epoch + 1, self.config.epochs);

            let current_lr = self.get_learning_rate(epoch, self.config.epochs);
            info!("Current learning rate: {:.6}", current_lr);

            let (total_loss, num_updates) = if self.config.use_parallel {
                self.epoch_cbow_parallel(data, neg_sampler.as_ref(), total_words, subsample_threshold, current_lr)
            } else {
                let mut total_loss = 0.0;
                let mut num_updates = 0;
                let mut batch_count = 0;
                let mut batch_loss = 0.0;
                let mut num_batches = 0;
                let mut accum = self.new_gradient_accumulator();

                for sentence in data.sentences.iter() {
                    for (target_idx, target_word) in sentence.iter().enumerate() {
                        if let Some(&target_id) = data.vocab.get(target_word) {
                            // Sub-sampling: skip very frequent words
                            if should_subsample(target_id, &data.word_freq, subsample_threshold, total_words, &mut rng) {
                                continue;
                            }

                            let start = target_idx.saturating_sub(self.config.context_window);
                            let end = std::cmp::min(target_idx + self.config.context_window + 1, sentence.len());

                            let mut context_ids = Vec::new();
                            for (context_idx, context_word) in sentence.iter().enumerate().skip(start).take(end - start) {
                                if context_idx != target_idx
                                    && let Some(&context_id) = data.vocab.get(context_word) {
                                        // Sub-sampling for context words too
                                        if should_subsample(context_id, &data.word_freq, subsample_threshold, total_words, &mut rng) {
                                            continue;
                                        }
                                        context_ids.push(context_id);
                                    }
                            }

                            if !context_ids.is_empty() {
                                // Get negative samples
                                let negative_samples = get_negative_samples(
                                    self.vocab_size,
                                    self.config.negative_samples,
                                    target_id,
                                    neg_sampler.as_ref(),
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

                (total_loss, num_updates)
            };

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

            // Save checkpoint if configured
            if let Some(interval) = self.config.checkpoint_interval {
                if (epoch + 1) % interval == 0 {
                    let dir = self.config.checkpoint_path.as_deref().unwrap_or(".");
                    let path = format!("{}/checkpoint_epoch_{}.json", dir, epoch + 1);
                    if let Err(e) = self.save_checkpoint(&path, epoch + 1, best_loss) {
                        tracing::warn!("Failed to save checkpoint: {}", e);
                    } else {
                        tracing::info!("Checkpoint saved: {}", path);
                    }
                }
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

    fn apply_merged_gradients(&mut self, merged: &HashMap<usize, Vec<f32>>, count: usize) {
        self.apply_gradient_batch(&mut merged.clone(), count);
    }

    /// Processes one epoch of Skip-gram training in parallel over sentences.
    /// Returns `(total_loss, num_updates)` and mutates `self.embeddings`.
    fn epoch_skipgram_parallel(
        &mut self,
        data: &TrainingData,
        neg_sampler: Option<&WeightedIndex<f32>>,
        total_words: f64,
        subsample_threshold: Option<f64>,
        current_lr: f32,
    ) -> (f32, usize) {
        let config = &self.config;
        let vocab = &data.vocab;
        let word_freq = &data.word_freq;
        let sentences = &data.sentences;
        let vocab_size = self.vocab_size;
        let neg_samples = config.negative_samples;
        let context_window = config.context_window;
        let embedding_dim = config.embedding_dim;

        let results: Vec<(f32, usize, HashMap<usize, Vec<f32>>)> = sentences.par_iter().map(|sentence| {
            let mut rng = StdRng::from_entropy();
            let mut accum = HashMap::new();
            let mut total_loss = 0.0f32;
            let mut num_updates = 0usize;

            for (target_idx, target_word) in sentence.iter().enumerate() {
                if let Some(&target_id) = vocab.get(target_word) {
                    if should_subsample(target_id, word_freq, subsample_threshold, total_words, &mut rng) {
                        continue;
                    }

                    let start = target_idx.saturating_sub(context_window);
                    let end = std::cmp::min(target_idx + context_window + 1, sentence.len());

                    let negative_samples = get_negative_samples(
                        vocab_size, neg_samples, target_id, neg_sampler, &mut rng
                    );

                    for (context_idx, context_word) in sentence.iter().enumerate().skip(start).take(end - start) {
                        if context_idx != target_idx && let Some(&context_id) = vocab.get(context_word) {
                            if should_subsample(context_id, word_freq, subsample_threshold, total_words, &mut rng) {
                                continue;
                            }

                            let loss = compute_skipgram_gradients_impl(
                                &self.embeddings, config, target_id, context_id,
                                &negative_samples, current_lr, &mut accum,
                            );
                            total_loss += loss;
                            num_updates += 1;
                        }
                    }
                }
            }
            (total_loss, num_updates, accum)
        }).collect();

        let mut merged = HashMap::new();
        let mut total_loss = 0.0f32;
        let mut total_updates = 0usize;
        for (loss, updates, accum) in results {
            total_loss += loss;
            total_updates += updates;
            for (word_id, grads) in accum {
                let entry = merged.entry(word_id).or_insert_with(|| vec![0.0; embedding_dim]);
                for (i, &g) in grads.iter().enumerate() {
                    entry[i] += g;
                }
            }
        }

        if total_updates > 0 {
            self.apply_gradient_batch(&mut merged, total_updates);
        }

        (total_loss, total_updates)
    }

    /// Processes one epoch of CBOW training in parallel over sentences.
    /// Returns `(total_loss, num_updates)` and mutates `self.embeddings`.
    fn epoch_cbow_parallel(
        &mut self,
        data: &TrainingData,
        neg_sampler: Option<&WeightedIndex<f32>>,
        total_words: f64,
        subsample_threshold: Option<f64>,
        current_lr: f32,
    ) -> (f32, usize) {
        let config = &self.config;
        let vocab = &data.vocab;
        let word_freq = &data.word_freq;
        let sentences = &data.sentences;
        let vocab_size = self.vocab_size;
        let neg_samples = config.negative_samples;
        let context_window = config.context_window;
        let embedding_dim = config.embedding_dim;

        let results: Vec<(f32, usize, HashMap<usize, Vec<f32>>)> = sentences.par_iter().map(|sentence| {
            let mut rng = StdRng::from_entropy();
            let mut accum = HashMap::new();
            let mut total_loss = 0.0f32;
            let mut num_updates = 0usize;

            for (target_idx, target_word) in sentence.iter().enumerate() {
                if let Some(&target_id) = vocab.get(target_word) {
                    if should_subsample(target_id, word_freq, subsample_threshold, total_words, &mut rng) {
                        continue;
                    }

                    let start = target_idx.saturating_sub(context_window);
                    let end = std::cmp::min(target_idx + context_window + 1, sentence.len());

                    let mut context_ids = Vec::new();
                    for (context_idx, context_word) in sentence.iter().enumerate().skip(start).take(end - start) {
                        if context_idx != target_idx && let Some(&context_id) = vocab.get(context_word) {
                            if should_subsample(context_id, word_freq, subsample_threshold, total_words, &mut rng) {
                                continue;
                            }
                            context_ids.push(context_id);
                        }
                    }

                    if context_ids.is_empty() {
                        continue;
                    }

                    let negative_samples = get_negative_samples(
                        vocab_size, neg_samples, target_id, neg_sampler, &mut rng
                    );

                    let loss = compute_cbow_gradients_impl(
                        &self.embeddings, config, target_id, &context_ids,
                        &negative_samples, current_lr, &mut accum,
                    );
                    total_loss += loss;
                    num_updates += 1;
                }
            }
            (total_loss, num_updates, accum)
        }).collect();

        let mut merged = HashMap::new();
        let mut total_loss = 0.0f32;
        let mut total_updates = 0usize;
        for (loss, updates, accum) in results {
            total_loss += loss;
            total_updates += updates;
            for (word_id, grads) in accum {
                let entry = merged.entry(word_id).or_insert_with(|| vec![0.0; embedding_dim]);
                for (i, &g) in grads.iter().enumerate() {
                    entry[i] += g;
                }
            }
        }

        if total_updates > 0 {
            self.apply_gradient_batch(&mut merged, total_updates);
        }

        (total_loss, total_updates)
    }

    fn compute_skipgram_gradients(
        &self,
        target_id: usize,
        context_id: usize,
        negative_samples: &[usize],
        learning_rate: f32,
        accum: &mut HashMap<usize, Vec<f32>>,
    ) -> f32 {
        compute_skipgram_gradients_impl(
            &self.embeddings, &self.config, target_id, context_id,
            negative_samples, learning_rate, accum,
        )
    }

    fn compute_cbow_gradients(
        &self,
        target_id: usize,
        context_ids: &[usize],
        negative_samples: &[usize],
        learning_rate: f32,
        accum: &mut HashMap<usize, Vec<f32>>,
    ) -> f32 {
        compute_cbow_gradients_impl(
            &self.embeddings, &self.config, target_id, context_ids,
            negative_samples, learning_rate, accum,
        )
    }

    pub(crate) fn get_learning_rate(&self, epoch: usize, total_epochs: usize) -> f32 {
        let base_lr = match self.config.lr_schedule {
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
        };

        // Apply linear warm-up
        if let Some(warmup) = self.config.warmup_epochs {
            if warmup > 0 && epoch < warmup {
                return base_lr * (epoch as f32 / warmup as f32);
            }
        }
        base_lr
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

/// Builds a unigram distribution raised to the 3/4 power for negative sampling.
fn build_negative_sampling_dist(word_freq: &[usize]) -> Vec<f32> {
    let mut dist: Vec<f32> = word_freq.iter().map(|&f| (f as f32).powf(0.75)).collect();
    let sum: f32 = dist.iter().sum();
    if sum > 0.0 {
        for p in &mut dist {
            *p /= sum;
        }
    }
    dist
}

/// Returns `true` if the word should be skipped based on sub-sampling (Mikolov et al.).
fn should_subsample(
    word_id: usize,
    word_freq: &[usize],
    threshold: Option<f64>,
    total_words: f64,
    rng: &mut impl Rng,
) -> bool {
    let Some(t) = threshold else { return false };
    if total_words == 0.0 {
        return false;
    }
    let f = word_freq.get(word_id).copied().unwrap_or(0) as f64 / total_words;
    if f <= t {
        return false;
    }
    let drop_prob = 1.0 - (t / f).sqrt();
    rng.gen_bool(drop_prob)
}

fn get_negative_samples<R: Rng>(
    vocab_size: usize,
    num_samples: usize,
    target_id: usize,
    neg_sampler: Option<&WeightedIndex<f32>>,
    rng: &mut R,
) -> Vec<usize> {
    let mut samples = Vec::new();

    if let Some(sampler) = neg_sampler {
        while samples.len() < num_samples {
            let candidate = rng.sample(sampler);
            if candidate != target_id && !samples.contains(&candidate) {
                samples.push(candidate);
            }
        }
    } else {
        let dist = Uniform::new(0, vocab_size);
        while samples.len() < num_samples {
            let candidate = rng.sample(dist);
            if candidate != target_id && !samples.contains(&candidate) {
                samples.push(candidate);
            }
        }
    }

    samples
}

/// Standalone skip-gram gradient computation (used by both sequential
/// and parallel training paths).
fn compute_skipgram_gradients_impl(
    embeddings: &ndarray::Array2<f32>,
    config: &crate::config::TrainingConfig,
    target_id: usize,
    context_id: usize,
    negative_samples: &[usize],
    learning_rate: f32,
    accum: &mut HashMap<usize, Vec<f32>>,
) -> f32 {
    let target_embedding: Vec<f32> = embeddings.row(target_id).to_vec();
    let context_embedding: Vec<f32> = embeddings.row(context_id).to_vec();

    let dot_product: f32 = target_embedding.iter().zip(context_embedding.iter()).map(|(&a, &b)| a * b).sum();
    let prob_positive = sigmoid(dot_product);
    let grad_positive = prob_positive - 1.0;

    for (i, (&t, &c)) in target_embedding.iter().zip(context_embedding.iter()).enumerate() {
        let l2_t = config.l2_regularization.map(|l2| -l2 as f32 * t).unwrap_or(0.0);
        let grad = learning_rate * grad_positive * c + l2_t;
        let grad = if let Some(max_norm) = config.gradient_clip {
            grad.clamp(-max_norm, max_norm)
        } else { grad };
        let entry = accum.entry(target_id).or_insert_with(|| vec![0.0; config.embedding_dim]);
        entry[i] += grad;

        let l2_c = config.l2_regularization.map(|l2| -l2 as f32 * c).unwrap_or(0.0);
        let grad_context = learning_rate * grad_positive * t + l2_c;
        let grad_context = if let Some(max_norm) = config.gradient_clip {
            grad_context.clamp(-max_norm, max_norm)
        } else { grad_context };
        let entry = accum.entry(context_id).or_insert_with(|| vec![0.0; config.embedding_dim]);
        entry[i] += grad_context;
    }

    let mut loss = -prob_positive.ln();
    for &neg_id in negative_samples {
        let neg_embedding: Vec<f32> = embeddings.row(neg_id).to_vec();
        let dot_product_neg: f32 = target_embedding.iter().zip(neg_embedding.iter()).map(|(&a, &b)| a * b).sum();
        let prob_negative = sigmoid(dot_product_neg);
        let grad_negative = prob_negative;
        loss += -(1.0 - prob_negative).ln();

        for (i, (&t, &n)) in target_embedding.iter().zip(neg_embedding.iter()).enumerate() {
            let l2_t = config.l2_regularization.map(|l2| -l2 as f32 * t).unwrap_or(0.0);
            let grad = learning_rate * grad_negative * n + l2_t;
            let grad = if let Some(max_norm) = config.gradient_clip {
                grad.clamp(-max_norm, max_norm)
            } else { grad };
            let entry = accum.entry(target_id).or_insert_with(|| vec![0.0; config.embedding_dim]);
            entry[i] += grad;

            let l2_n = config.l2_regularization.map(|l2| -l2 as f32 * n).unwrap_or(0.0);
            let grad_neg = learning_rate * grad_negative * t + l2_n;
            let grad_neg = if let Some(max_norm) = config.gradient_clip {
                grad_neg.clamp(-max_norm, max_norm)
            } else { grad_neg };
            let entry = accum.entry(neg_id).or_insert_with(|| vec![0.0; config.embedding_dim]);
            entry[i] += grad_neg;
        }
    }
    loss
}

/// Standalone CBOW gradient computation (used by both sequential
/// and parallel training paths).
fn compute_cbow_gradients_impl(
    embeddings: &ndarray::Array2<f32>,
    config: &crate::config::TrainingConfig,
    target_id: usize,
    context_ids: &[usize],
    negative_samples: &[usize],
    learning_rate: f32,
    accum: &mut HashMap<usize, Vec<f32>>,
) -> f32 {
    let mut context_vector = ndarray::Array::zeros(config.embedding_dim);
    let mut context_embeddings: Vec<Vec<f32>> = Vec::new();

    for &context_id in context_ids {
        let context_embedding: Vec<f32> = embeddings.row(context_id).to_vec();
        context_embeddings.push(context_embedding.clone());
        let context_arr = ndarray::Array::from_shape_vec((config.embedding_dim,), context_embedding).unwrap();
        context_vector += &context_arr;
    }
    context_vector /= context_ids.len() as f32;

    let target_embedding: Vec<f32> = embeddings.row(target_id).to_vec();
    let target_arr = ndarray::Array::from_shape_vec((config.embedding_dim,), target_embedding.clone()).unwrap();
    let dot_product = context_vector.dot(&target_arr);
    let prob_positive = sigmoid(dot_product);
    let grad_positive = prob_positive - 1.0;

    for (context_idx, &context_id) in context_ids.iter().enumerate() {
        let _context_embedding = &context_embeddings[context_idx];
        for (i, &target_val) in target_embedding.iter().enumerate() {
            let weight = embeddings[[context_id, i]];
            let l2 = config.l2_regularization.map(|l2| -l2 as f32 * weight).unwrap_or(0.0);
            let grad = learning_rate * grad_positive * target_val + l2;
            let grad = if let Some(max_norm) = config.gradient_clip {
                grad.clamp(-max_norm, max_norm)
            } else { grad };
            let entry = accum.entry(context_id).or_insert_with(|| vec![0.0; config.embedding_dim]);
            entry[i] += grad;
        }
    }

    for (i, &context_val) in context_vector.iter().enumerate() {
        let l2 = config.l2_regularization.map(|l2| -l2 as f32 * target_embedding[i]).unwrap_or(0.0);
        let grad = learning_rate * grad_positive * context_val + l2;
        let grad = if let Some(max_norm) = config.gradient_clip {
            grad.clamp(-max_norm, max_norm)
        } else { grad };
        let entry = accum.entry(target_id).or_insert_with(|| vec![0.0; config.embedding_dim]);
        entry[i] += grad;
    }

    let mut loss = -prob_positive.ln();
    for &neg_id in negative_samples {
        let neg_embedding: Vec<f32> = embeddings.row(neg_id).to_vec();
        let neg_arr = ndarray::Array::from_shape_vec((config.embedding_dim,), neg_embedding.clone()).unwrap();
        let dot_product_neg = context_vector.dot(&neg_arr);
        let prob_negative = sigmoid(dot_product_neg);
        let grad_negative = prob_negative;
        loss += -(1.0 - prob_negative).ln();

        for &context_id in context_ids {
            for (i, &neg_val) in neg_embedding.iter().enumerate() {
                let weight = embeddings[[context_id, i]];
                let l2 = config.l2_regularization.map(|l2| -l2 as f32 * weight).unwrap_or(0.0);
                let grad = learning_rate * grad_negative * neg_val + l2;
                let grad = if let Some(max_norm) = config.gradient_clip {
                    grad.clamp(-max_norm, max_norm)
                } else { grad };
                let entry = accum.entry(context_id).or_insert_with(|| vec![0.0; config.embedding_dim]);
                entry[i] += grad;
            }
        }

        for (i, &context_val) in context_vector.iter().enumerate() {
            let l2 = config.l2_regularization.map(|l2| -l2 as f32 * neg_embedding[i]).unwrap_or(0.0);
            let grad = learning_rate * grad_negative * context_val + l2;
            let grad = if let Some(max_norm) = config.gradient_clip {
                grad.clamp(-max_norm, max_norm)
            } else { grad };
            let entry = accum.entry(neg_id).or_insert_with(|| vec![0.0; config.embedding_dim]);
            entry[i] += grad;
        }
    }
    loss
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
