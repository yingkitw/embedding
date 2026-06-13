use ndarray::{Array, Array1, Array2};
use std::collections::HashMap;
use crate::{EmbeddingModel, TrainingData};

/// Combines text embeddings with auxiliary vectors (e.g., image features).
#[derive(Debug, Clone)]
pub struct MultimodalFusion {
    pub text_dim: usize,
    pub aux_dim: usize,
    pub fused_dim: usize,
}

impl MultimodalFusion {
    /// Creates a fusion module with the given dimensions.
    /// `fused_dim` is the output dimension after projection.
    pub fn new(text_dim: usize, aux_dim: usize, fused_dim: usize) -> Self {
        Self {
            text_dim,
            aux_dim,
            fused_dim,
        }
    }

    /// Concatenates a text embedding with an auxiliary vector.
    pub fn concatenate(&self, text: &Array1<f32>, aux: &Array1<f32>) -> Array1<f32> {
        let mut result = Array::zeros(self.text_dim + self.aux_dim);
        result.slice_mut(ndarray::s![..self.text_dim]).assign(text);
        result.slice_mut(ndarray::s![self.text_dim..]).assign(aux);
        result
    }

    /// Weighted average of text and auxiliary embeddings.
    /// Both vectors must have the same dimensionality.
    pub fn weighted_average(&self, text: &Array1<f32>, aux: &Array1<f32>, text_weight: f32) -> Option<Array1<f32>> {
        if text.len() != aux.len() {
            return None;
        }
        let aux_weight = 1.0 - text_weight;
        Some(text * text_weight + aux * aux_weight)
    }

    /// Attention-based fusion: computes a scalar attention weight from
    /// the dot product of text and auxiliary vectors, then blends them.
    /// Vectors must have the same dimensionality.
    pub fn attention_fusion(&self, text: &Array1<f32>, aux: &Array1<f32>) -> Option<Array1<f32>> {
        if text.len() != aux.len() {
            return None;
        }
        let dot: f32 = text.iter().zip(aux.iter()).map(|(&a, &b)| a * b).sum();
        let norm_t = text.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_a = aux.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm_t == 0.0 || norm_a == 0.0 {
            return Some(text.clone());
        }
        let raw_attn = (dot / (norm_t * norm_a)).tanh();
        let attn = 0.5 + 0.5 * raw_attn; // scale to [0, 1]
        Some(text * attn + aux * (1.0 - attn))
    }

    /// Projects both modalities into a shared space and then averages.
    /// Returns a vector of length `fused_dim`.
    pub fn project_and_fuse(
        &self,
        text: &Array1<f32>,
        aux: &Array1<f32>,
        text_proj: &Array2<f32>,
        aux_proj: &Array2<f32>,
    ) -> Option<Array1<f32>> {
        if text_proj.shape()[1] != self.fused_dim || aux_proj.shape()[1] != self.fused_dim {
            return None;
        }
        let text_p = text.dot(text_proj);
        let aux_p = aux.dot(aux_proj);
        Some(&(text_p + aux_p) / 2.0)
    }

    /// Computes cross-modal cosine similarity between text and auxiliary vectors.
    pub fn cross_modal_similarity(text: &Array1<f32>, aux: &Array1<f32>) -> f32 {
        if text.len() != aux.len() {
            return 0.0;
        }
        let dot: f32 = text.iter().zip(aux.iter()).map(|(&a, &b)| a * b).sum();
        let norm_t = text.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_a = aux.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm_t == 0.0 || norm_a == 0.0 {
            0.0
        } else {
            dot / (norm_t * norm_a)
        }
    }
}

/// Linear alignment of embeddings across languages using a projection matrix.
#[derive(Debug, Clone)]
pub struct CrossLingualAligner {
    pub projection: Array2<f32>,
}

impl CrossLingualAligner {
    /// Creates an aligner with an identity-initialized projection matrix.
    pub fn new(dim: usize) -> Self {
        let mut proj = Array::zeros((dim, dim));
        for i in 0..dim {
            proj[[i, i]] = 1.0;
        }
        Self { projection: proj }
    }

    /// Projects a source-language embedding into the target-language space.
    pub fn align(&self, embedding: &Array1<f32>) -> Array1<f32> {
        self.projection.dot(embedding)
    }

    /// Trains the projection matrix from a list of translation pairs using
    /// least-squares via stochastic gradient descent.
    pub fn train_from_dictionary(
        &mut self,
        pairs: &[(Array1<f32>, Array1<f32>)],
        epochs: usize,
        learning_rate: f32,
    ) {
        let dim = self.projection.nrows();
        for _ in 0..epochs {
            for (src, tgt) in pairs {
                let pred = self.projection.dot(src);
                let error = &pred - tgt;
                for i in 0..dim {
                    let grad = error[i] * src;
                    let mut row = self.projection.row_mut(i);
                    row -= &(grad * learning_rate);
                }
            }
        }
    }
}

/// Adapts a general embedding model to a specific domain by fine-tuning
/// on domain sentences with an optional domain-vocabulary boost.
pub struct DomainAdapter;

impl DomainAdapter {
    /// Fine-tunes a model on domain-specific sentences for a small number of epochs.
    pub fn adapt(
        model: &mut EmbeddingModel,
        data: &mut TrainingData,
        domain_sentences: &[Vec<String>],
        epochs: usize,
    ) -> Result<(), String> {
        let mut domain_data = TrainingData {
            sentences: domain_sentences.to_vec(),
            vocab: data.vocab.clone(),
            reverse_vocab: data.reverse_vocab.clone(),
            word_freq: data.word_freq.clone(),
        };
        // Incremental vocab update for any new domain words
        let domain_words: Vec<String> = domain_sentences
            .iter()
            .flat_map(|s| s.iter().cloned())
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();
        model.incremental_vocab_update(&domain_words, &mut domain_data)?;

        let original_epochs = model.config.epochs;
        model.config.epochs = epochs;
        model.train(&domain_data)?;
        model.config.epochs = original_epochs;

        // Update caller's data to reflect new vocabulary
        *data = domain_data;
        Ok(())
    }
}

/// Produces document-level embeddings by aggregating sentence embeddings.
pub struct DocumentEmbedder;

impl DocumentEmbedder {
    /// Returns a document vector by mean-pooling sentence embeddings.
    pub fn embed_document(
        model: &EmbeddingModel,
        data: &TrainingData,
        sentences: &[Vec<String>],
    ) -> Option<Array1<f32>> {
        if sentences.is_empty() {
            return None;
        }
        let mut sum = Array::zeros(model.config.embedding_dim);
        let mut count = 0usize;
        for sentence in sentences {
            if let Some(emb) = model.sentence_embedding(sentence, data) {
                sum += &emb;
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        Some(&sum / (count as f32))
    }
}

/// FastText-style subword embeddings using character n-grams.
///
/// Represents words as the sum of their character n-gram vectors,
/// enabling embeddings for out-of-vocabulary words.
pub struct SubwordEmbedder {
    pub min_n: usize,
    pub max_n: usize,
}

impl SubwordEmbedder {
    /// Creates a subword embedder with the given n-gram range.
    pub fn new(min_n: usize, max_n: usize) -> Self {
        Self { min_n, max_n }
    }

    /// Extracts character n-grams from a word with boundary markers.
    pub fn ngrams(&self, word: &str) -> Vec<String> {
        let bounded = format!("<{}>", word);
        let chars: Vec<char> = bounded.chars().collect();
        let mut result = Vec::new();
        for n in self.min_n..=self.max_n {
            if n > chars.len() {
                continue;
            }
            for window in chars.windows(n) {
                result.push(window.iter().collect());
            }
        }
        result
    }

    /// Builds an embedding for a word by summing n-gram vectors.
    pub fn embed(&self, word: &str, ngram_vectors: &HashMap<String, Array1<f32>>) -> Option<Array1<f32>> {
        let grams = self.ngrams(word);
        if grams.is_empty() {
            return None;
        }
        let mut sum: Option<Array1<f32>> = None;
        let mut count = 0usize;
        for gram in grams {
            if let Some(vec) = ngram_vectors.get(&gram) {
                sum = Some(match sum {
                    Some(s) => s + vec,
                    None => vec.clone(),
                });
                count += 1;
            }
        }
        sum.map(|s| s / (count.max(1) as f32))
    }
}

/// Zero-shot transfer learning via nearest-neighbor class prototype matching.
pub struct ZeroShotTransfer;

impl ZeroShotTransfer {
    /// Given class prototypes (label -> prototype vector), assigns the nearest
    /// prototype to each query embedding by cosine similarity.
    pub fn classify(
        query: &Array1<f32>,
        class_prototypes: &HashMap<String, Array1<f32>>,
    ) -> Option<(String, f32)> {
        let mut best_label = None;
        let mut best_sim = f32::NEG_INFINITY;
        for (label, proto) in class_prototypes {
            let dot: f32 = query.iter().zip(proto.iter()).map(|(&a, &b)| a * b).sum();
            let norm_q = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm_p = proto.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_q > 0.0 && norm_p > 0.0 {
                let sim = dot / (norm_q * norm_p);
                if sim > best_sim {
                    best_sim = sim;
                    best_label = Some(label.clone());
                }
            }
        }
        best_label.map(|l| (l, best_sim))
    }
}
