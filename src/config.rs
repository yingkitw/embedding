use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::prelude::SliceRandom;
use crate::text::{load_text_data, TextProcessor};

/// Default embedding dimension.
pub const DEFAULT_EMBEDDING_DIM: usize = 300;
/// Default learning rate.
pub const DEFAULT_LEARNING_RATE: f64 = 0.025;
/// Default number of training epochs.
pub const DEFAULT_EPOCHS: usize = 10;
/// Default mini-batch size.
pub const DEFAULT_BATCH_SIZE: usize = 32;
/// Default context window size.
pub const DEFAULT_CONTEXT_WINDOW: usize = 5;
/// Default number of negative samples.
pub const DEFAULT_NEGATIVE_SAMPLES: usize = 5;
/// Default validation ratio (0.0 = no validation).
pub const DEFAULT_VALIDATION_RATIO: f64 = 0.0;

/// Training hyperparameters for embedding models.
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
    pub gradient_clip: Option<f32>,
    pub validation_ratio: Option<f64>,
    /// Sub-sampling threshold for frequent words (Mikolov et al.).
    /// `None` disables sub-sampling. Typical value: `1e-5`.
    pub subsample_threshold: Option<f64>,
    /// If `true`, sample negative examples using the unigram distribution
    /// raised to the 3/4 power instead of uniform random.
    pub use_unigram_negative_sampling: bool,
    /// Number of epochs for linear LR warm-up. `None` disables warm-up.
    pub warmup_epochs: Option<usize>,
    /// Save a checkpoint every N epochs. `None` disables checkpointing.
    pub checkpoint_interval: Option<usize>,
    /// Directory to write checkpoint files. Defaults to current directory.
    pub checkpoint_path: Option<String>,
    /// If `true`, process sentences in parallel during training.
    pub use_parallel: bool,
}

impl TrainingConfig {
    /// Creates a new [`TrainingConfig`] with sensible defaults.
    ///
    /// # Example
    /// ```rust
    /// use embedding::{TrainingConfig, ModelType};
    /// let config = TrainingConfig::new(ModelType::SkipGram);
    /// ```
    pub fn new(model_type: ModelType) -> Self {
        Self {
            embedding_dim: DEFAULT_EMBEDDING_DIM,
            learning_rate: DEFAULT_LEARNING_RATE,
            epochs: DEFAULT_EPOCHS,
            batch_size: DEFAULT_BATCH_SIZE,
            context_window: DEFAULT_CONTEXT_WINDOW,
            negative_samples: DEFAULT_NEGATIVE_SAMPLES,
            model_type,
            lr_schedule: LearningRateSchedule::Constant,
            early_stopping: None,
            l2_regularization: None,
            gradient_clip: None,
            validation_ratio: None,
            subsample_threshold: None,
            use_unigram_negative_sampling: true,
            warmup_epochs: None,
            checkpoint_interval: None,
            checkpoint_path: None,
            use_parallel: false,
        }
    }

    /// Fluent setter for embedding dimension.
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Fluent setter for learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Fluent setter for number of epochs.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Fluent setter for batch size.
    pub fn with_batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
        self
    }

    /// Fluent setter for context window.
    pub fn with_window(mut self, window: usize) -> Self {
        self.context_window = window;
        self
    }

    /// Fluent setter for negative samples.
    pub fn with_negative_samples(mut self, ns: usize) -> Self {
        self.negative_samples = ns;
        self
    }

    /// Fluent setter for learning rate schedule.
    pub fn with_lr_schedule(mut self, schedule: LearningRateSchedule) -> Self {
        self.lr_schedule = schedule;
        self
    }

    /// Fluent setter for early stopping.
    pub fn with_early_stopping(mut self, patience: usize, min_delta: f64) -> Self {
        self.early_stopping = Some(EarlyStoppingConfig { patience, min_delta });
        self
    }

    /// Fluent setter for L2 regularization.
    pub fn with_l2_regularization(mut self, lambda: f64) -> Self {
        self.l2_regularization = Some(lambda);
        self
    }

    /// Fluent setter for gradient clip.
    pub fn with_gradient_clip(mut self, max_norm: f32) -> Self {
        self.gradient_clip = Some(max_norm);
        self
    }

    /// Fluent setter for validation ratio.
    pub fn with_validation_ratio(mut self, ratio: f64) -> Self {
        self.validation_ratio = Some(ratio);
        self
    }

    /// Fluent setter for sub-sampling threshold (`None` disables sub-sampling).
    pub fn with_subsample_threshold(mut self, threshold: Option<f64>) -> Self {
        self.subsample_threshold = threshold;
        self
    }

    /// Fluent setter for unigram negative sampling.
    pub fn with_unigram_negative_sampling(mut self, enabled: bool) -> Self {
        self.use_unigram_negative_sampling = enabled;
        self
    }

    /// Fluent setter for LR warm-up epochs (`None` disables warm-up).
    pub fn with_warmup_epochs(mut self, epochs: Option<usize>) -> Self {
        self.warmup_epochs = epochs;
        self
    }

    /// Fluent setter for checkpoint interval (`None` disables checkpointing).
    pub fn with_checkpoint_interval(mut self, interval: Option<usize>) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Fluent setter for checkpoint directory.
    pub fn with_checkpoint_path(mut self, path: Option<String>) -> Self {
        self.checkpoint_path = path;
        self
    }

    /// Fluent setter for parallel training.
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.use_parallel = enabled;
        self
    }
}

/// Learning rate schedule variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant,
    Exponential { decay_rate: f64 },
    Step { step_size: usize, gamma: f64 },
    Cosine { t_max: usize },
}

/// Configuration for early stopping during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f64,
}

/// Supported embedding model architectures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelType {
    SkipGram,
    Cbow,
}

/// Container for tokenized sentences and the vocabulary mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    pub sentences: Vec<Vec<String>>,
    pub vocab: HashMap<String, usize>,
    pub reverse_vocab: Vec<String>,
    /// Per-vocab-ID word frequencies used for negative-sampling and sub-sampling.
    pub word_freq: Vec<usize>,
}

impl TrainingData {
    /// Creates [`TrainingData`] from raw text by tokenizing and building the vocabulary.
    ///
    /// # Example
    /// ```rust
    /// use embedding::TrainingData;
    /// let data = TrainingData::from_text("the cat sat on the mat");
    /// ```
    pub fn from_text(text: &str) -> Self {
        let sentences = load_text_data(text);
        let (vocab, reverse_vocab, word_freq) = crate::text::build_vocab_with_freq(&sentences);
        Self { sentences, vocab, reverse_vocab, word_freq }
    }

    /// Creates [`TrainingData`] from a file by reading, tokenizing, and building the vocabulary.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let sentences = load_text_data(&content);
        let (vocab, reverse_vocab, word_freq) = crate::text::build_vocab_with_freq(&sentences);
        Ok(Self { sentences, vocab, reverse_vocab, word_freq })
    }

    /// Total number of word occurrences across all sentences.
    pub fn total_word_count(&self) -> usize {
        self.word_freq.iter().sum()
    }
}

/// Utility for batching and streaming sentence data.
#[derive(Debug, Clone)]
pub struct DataLoader {
    pub batch_size: usize,
    pub shuffle: bool,
    pub file_path: Option<String>,
}

impl DataLoader {
    /// Creates a new data loader with the given batch size and options.
    pub fn new(batch_size: usize, shuffle: bool) -> Self {
        Self {
            batch_size,
            shuffle,
            file_path: None,
        }
    }
    
    /// Sets the file path for lazy loading.
    pub fn set_file_path(&mut self, path: String) {
        self.file_path = Some(path);
    }
    
    /// Groups sentences into fixed-size batches.
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
    
    /// Loads sentences from a file.
    pub fn load_lazily(&self, file_path: &str) -> Result<Vec<Vec<String>>, String> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(file_path).map_err(|e| e.to_string())?;
        let mut content = String::new();
        file.read_to_string(&mut content).map_err(|e| e.to_string())?;

        Ok(load_text_data(&content))
    }

    /// Returns a lazy iterator over sentences from a file.
    pub fn stream_sentences(&self, file_path: &str) -> Result<Box<dyn Iterator<Item = Vec<String>>>, String> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(file_path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let processor = TextProcessor::default();

        let iter = reader.lines().filter_map(move |line| {
            let line = line.ok()?;
            let sentences = processor.process_text(&line);
            if sentences.is_empty() {
                None
            } else {
                Some(sentences.into_iter().flatten().collect::<Vec<String>>())
            }
        });

        Ok(Box::new(iter))
    }
}
