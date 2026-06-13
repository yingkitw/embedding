use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;

/// Base for sinusoidal position encoding (from "Attention Is All You Need").
const POSITION_ENCODING_BASE: f32 = 10000.0;
/// Small epsilon to prevent division by zero in layer normalization.
const LAYER_NORM_EPSILON: f32 = 1e-5;

/// A lightweight transformer encoder for producing contextualized embeddings.
///
/// This is not a full BERT-scale model, but a small multi-layer transformer
/// suitable for producing context-aware sentence embeddings from word vectors.
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    pub n_layers: usize,
    pub n_heads: usize,
    pub dim: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
    pub attention_weights: Vec<Array3<f32>>,
    pub feed_forward_weights: Vec<(Array2<f32>, Array1<f32>)>,
    pub layer_norms: Vec<(Array1<f32>, Array1<f32>)>,
}

impl TransformerEncoder {
    /// Creates a new transformer encoder with Xavier-initialized weights.
    pub fn new(n_layers: usize, n_heads: usize, dim: usize, ff_dim: usize, max_seq_len: usize) -> Self {
        assert_eq!(dim % n_heads, 0, "dim must be divisible by n_heads");
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (dim as f32).sqrt();

        let mut attention_weights = Vec::with_capacity(n_layers);
        let mut feed_forward_weights = Vec::with_capacity(n_layers);
        let mut layer_norms = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            // Query, Key, Value projection weights (dim x dim)
            let w_qkv = Array2::from_shape_fn((dim, dim), |_| rng.gen_range(-0.5..0.5) * scale);
            let w_out = Array2::from_shape_fn((dim, dim), |_| rng.gen_range(-0.5..0.5) * scale);
            attention_weights.push(
                ndarray::stack![Axis(0), w_qkv.view(), w_out.view()]
                    .into_shape((2, dim, dim))
                    .unwrap(),
            );

            // Feed-forward: dim -> ff_dim -> dim
            let w1 = Array2::from_shape_fn((dim, ff_dim), |_| rng.gen_range(-0.5..0.5) * scale);
            let b1 = Array1::zeros(ff_dim);
            feed_forward_weights.push((w1, b1));

            // Layer norm gamma/beta
            let gamma = Array1::ones(dim);
            let beta = Array1::zeros(dim);
            layer_norms.push((gamma, beta));
        }

        Self {
            n_layers,
            n_heads,
            dim,
            ff_dim,
            max_seq_len,
            attention_weights,
            feed_forward_weights,
            layer_norms,
        }
    }

    /// Computes sinusoidal position encodings for a sequence of length `seq_len`.
    pub fn position_encoding(&self, seq_len: usize) -> Array2<f32> {
        let mut pe = Array2::zeros((seq_len, self.dim));
        for pos in 0..seq_len {
            for i in (0..self.dim).step_by(2) {
                let angle = pos as f32 / (POSITION_ENCODING_BASE.powf(i as f32 / self.dim as f32));
                pe[[pos, i]] = angle.sin();
                if i + 1 < self.dim {
                    pe[[pos, i + 1]] = angle.cos();
                }
            }
        }
        pe
    }

    /// Encodes a sequence of token embeddings into contextualized representations.
    ///
    /// `tokens` has shape `(seq_len, dim)`.
    /// Returns an array of shape `(seq_len, dim)`.
    pub fn encode_sequence(&self, tokens: &Array2<f32>) -> Array2<f32> {
        let seq_len = tokens.nrows();
        let pe = self.position_encoding(seq_len);
        let mut x = tokens + &pe;

        for layer in 0..self.n_layers {
            // Self-attention
            let attn_out = self.multi_head_attention(&x, layer);
            x = &x + &attn_out;
            x = self.layer_norm(&x, &self.layer_norms[layer].0, &self.layer_norms[layer].1);

            // Feed-forward
            let ff_out = self.feed_forward(&x, layer);
            x = &x + &ff_out;
            x = self.layer_norm(&x, &self.layer_norms[layer].0, &self.layer_norms[layer].1);
        }

        x
    }

    fn multi_head_attention(&self, x: &Array2<f32>, layer: usize) -> Array2<f32> {
        let _seq_len = x.nrows();
        let head_dim = self.dim / self.n_heads;
        let weights = &self.attention_weights[layer];
        let w_qkv = weights.slice(ndarray::s![0, .., ..]);
        let w_out = weights.slice(ndarray::s![1, .., ..]);

        // Project to Q, K, V
        let qkv: Array2<f32> = x.dot(&w_qkv.t());

        let mut attn_outputs: Vec<Array2<f32>> = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            let q = qkv.slice(ndarray::s![.., start..end]);
            let k = qkv.slice(ndarray::s![.., start..end]);
            let v = qkv.slice(ndarray::s![.., start..end]);

            let scores = q.dot(&k.t()) / (head_dim as f32).sqrt();
            let mut attn_weights = scores.mapv(|s: f32| s.exp());
            for r in 0..attn_weights.nrows() {
                let sum: f32 = (0..attn_weights.ncols()).map(|c| attn_weights[[r, c]]).sum();
                if sum > 0.0 {
                    for c in 0..attn_weights.ncols() {
                        attn_weights[[r, c]] /= sum;
                    }
                }
            }
            attn_outputs.push(attn_weights.dot(&v));
        }

        let concatenated = ndarray::concatenate(Axis(1), &attn_outputs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        concatenated.dot(&w_out.t())
    }

    fn feed_forward(&self, x: &Array2<f32>, layer: usize) -> Array2<f32> {
        let (w, b) = &self.feed_forward_weights[layer];
        let hidden = x.dot(w);
        let hidden = &hidden + b;
        // ReLU activation
        let activated = hidden.mapv(|v| v.max(0.0));
        // Second linear projection back to dim
        let w2 = Array2::from_shape_fn((self.ff_dim, self.dim), |_| {
            let mut rng = rand::thread_rng();
            rng.gen_range(-0.5..0.5) / (self.ff_dim as f32).sqrt()
        });
        activated.dot(&w2)
    }

    fn layer_norm(&self, x: &Array2<f32>, gamma: &Array1<f32>, beta: &Array1<f32>) -> Array2<f32> {
        let mut out = Array2::zeros(x.raw_dim());
        for (mut row_out, row_x) in out.rows_mut().into_iter().zip(x.rows()) {
            let mean = row_x.mean().unwrap_or(0.0);
            let var = row_x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / row_x.len() as f32;
            let std = (var + LAYER_NORM_EPSILON).sqrt();
            for (i, &v) in row_x.iter().enumerate() {
                row_out[i] = ((v - mean) / std) * gamma[i] + beta[i];
            }
        }
        out
    }
}
