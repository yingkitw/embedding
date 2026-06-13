use ndarray::{Array1, Array2};

/// Abstraction over compute backends for matrix operations.
///
/// The library ships with a CPU backend by default. GPU acceleration
/// can be enabled via the `gpu` feature flag (experimental).
pub trait Backend {
    /// Initializes an embedding matrix with the given shape.
    fn init_embeddings(&self, vocab_size: usize, dim: usize) -> Array2<f32>;

    /// Computes the dot product of two vectors.
    fn dot(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32;

    /// Adds vector `b` scaled by `scale` into vector `a` in-place.
    fn add_scaled(&self, a: &mut Array1<f32>, b: &Array1<f32>, scale: f32);

    /// Returns the backend name for diagnostics.
    fn name(&self) -> &'static str;
}

/// CPU backend using ndarray (default).
#[derive(Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    fn init_embeddings(&self, vocab_size: usize, dim: usize) -> Array2<f32> {
        use ndarray::Array;
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (dim as f32).sqrt();
        Array::from_shape_fn((vocab_size, dim), |_| rng.gen_range(-0.5..0.5) * scale)
    }

    fn dot(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn add_scaled(&self, a: &mut Array1<f32>, b: &Array1<f32>, scale: f32) {
        for (ai, bi) in a.iter_mut().zip(b.iter()) {
            *ai += bi * scale;
        }
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

/// Returns the default backend (CPU).
pub fn default_backend() -> Box<dyn Backend> {
    Box::new(CpuBackend::new())
}
