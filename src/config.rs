use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::prelude::SliceRandom;
use crate::text::{load_text_data, TextProcessor};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
