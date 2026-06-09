use clap::{Parser, Subcommand};
use embedding::*;
use std::fs;
use std::path::Path;
use tracing::{info, error};

#[derive(Parser)]
#[command(name = "embedding-train")]
#[command(about = "A CLI tool for training word embeddings from scratch")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train embeddings from text data
    Train {
        /// Input text file path
        #[arg(short, long)]
        input: String,

        /// Output model file path
        #[arg(short, long)]
        output: String,

        /// Output embeddings file path
        #[arg(short, long)]
        embeddings: String,

        /// Config JSON file path (overrides other flags)
        #[arg(short, long)]
        config: Option<String>,

        /// Embedding dimension
        #[arg(short, long, default_value = "300")]
        dim: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.025")]
        learning_rate: f64,

        /// Number of training epochs
        #[arg(short, long, default_value = "10")]
        epochs: usize,

        /// Batch size
        #[arg(short, long, default_value = "32")]
        batch_size: usize,

        /// Context window size
        #[arg(short, long, default_value = "5")]
        window: usize,

        /// Number of negative samples
        #[arg(short, long, default_value = "5")]
        negative_samples: usize,

        /// Model type: skipgram, cbow, or sentencebert
        #[arg(short, long, default_value = "skipgram")]
        model_type: String,
    },
    
    /// Calculate similarity between two words
    Similarity {
        /// First word
        word1: String,
        
        /// Second word
        word2: String,
        
        /// Model file path
        #[arg(short, long)]
        model: String,
        
        /// Vocabulary file path
        #[arg(short, long)]
        vocab: String,
    },
    
    /// Load and inspect a trained model
    Info {
        /// Model file path
        #[arg(short, long)]
        model: String,
        
        /// Vocabulary file path
        #[arg(short, long)]
        vocab: String,
    },
    
    /// Export embeddings to different formats
    Export {
        /// Model file path
        #[arg(short, long)]
        model: String,
        
        /// Vocabulary file path
        #[arg(short, long)]
        vocab: String,
        
        /// Output file path
        #[arg(short, long)]
        output: String,
        
        /// Export format: text, json, or bin
        #[arg(short, long, default_value = "text")]
        format: String,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Train {
            input,
            output,
            embeddings,
            config: config_path,
            dim,
            learning_rate,
            epochs,
            batch_size,
            window,
            negative_samples,
            model_type,
        } => {
            info!("Starting embedding training...");

            // Load and prepare data
            let text = fs::read_to_string(&input)
                .unwrap_or_else(|_| {
                    error!("Failed to read input file: {}", input);
                    std::process::exit(1);
                });

            let sentences = load_text_data(&text);
            info!("Loaded {} sentences", sentences.len());

            let (vocab, reverse_vocab) = build_vocab(&sentences);
            info!("Built vocabulary with {} words", vocab.len());

            let training_data = TrainingData {
                sentences,
                vocab,
                reverse_vocab,
            };

            // Parse model type
            let model_type = match model_type.as_str() {
                "skipgram" => ModelType::SkipGram,
                "cbow" => ModelType::Cbow,
                "sentencebert" => ModelType::SentenceBERT,
                _ => {
                    error!("Unknown model type: {}. Use skipgram, cbow, or sentencebert", model_type);
                    std::process::exit(1);
                }
            };

            // Create training config from CLI or JSON file
            let config = if let Some(path) = config_path {
                let config_json = fs::read_to_string(&path)
                    .unwrap_or_else(|_| {
                        error!("Failed to read config file: {}", path);
                        std::process::exit(1);
                    });
                serde_json::from_str(&config_json)
                    .unwrap_or_else(|e| {
                        error!("Failed to parse config file: {}", e);
                        std::process::exit(1);
                    })
            } else {
                TrainingConfig {
                    embedding_dim: dim,
                    learning_rate,
                    epochs,
                    batch_size,
                    context_window: window,
                    negative_samples,
                    model_type,
                    lr_schedule: LearningRateSchedule::Constant,
                    early_stopping: None,
                    l2_regularization: None,
                    dropout_rate: None,
                }
            };
            
            // Initialize and train model
            let mut model = EmbeddingModel::new(config.clone(), training_data.vocab.len());
            info!("Training model with config: {:?}", config);
            
            if let Err(e) = model.train(&training_data) {
                error!("Training failed: {}", e);
                std::process::exit(1);
            }
            
            // Save model and embeddings
            let model_data = serde_json::to_string(&(&model, &training_data))
                .unwrap_or_else(|_| {
                    error!("Failed to serialize model");
                    std::process::exit(1);
                });
            
            fs::write(&output, model_data)
                .unwrap_or_else(|_| {
                    error!("Failed to save model to: {}", output);
                    std::process::exit(1);
                });
            
            if let Err(e) = model.save_embeddings(&embeddings, &training_data) {
                error!("Failed to save embeddings: {}", e);
                std::process::exit(1);
            }
            
            info!("Training completed successfully!");
            info!("Model saved to: {}", output);
            info!("Embeddings saved to: {}", embeddings);
        }
        
        Commands::Similarity { word1, word2, model, vocab: _vocab } => {
            info!("Calculating similarity between '{}' and '{}'", word1, word2);
            
            let model_data = fs::read_to_string(&model)
                .unwrap_or_else(|_| {
                    error!("Failed to read model file: {}", model);
                    std::process::exit(1);
                });
            
            let (model, training_data): (EmbeddingModel, TrainingData) = serde_json::from_str(&model_data)
                .unwrap_or_else(|_| {
                    error!("Failed to deserialize model");
                    std::process::exit(1);
                });
            
            if let Some(similarity) = model.similarity(&word1, &word2, &training_data) {
                info!("Similarity: {:.4}", similarity);
                println!("Similarity between '{}' and '{}': {:.4}", word1, word2, similarity);
            } else {
                error!("One or both words not found in vocabulary");
                std::process::exit(1);
            }
        }
        
        Commands::Info { model, vocab: _vocab } => {
            info!("Inspecting model...");
            
            let model_data = fs::read_to_string(&model)
                .unwrap_or_else(|_| {
                    error!("Failed to read model file: {}", model);
                    std::process::exit(1);
                });
            
            let (model, training_data): (EmbeddingModel, TrainingData) = serde_json::from_str(&model_data)
                .unwrap_or_else(|_| {
                    error!("Failed to deserialize model");
                    std::process::exit(1);
                });
            
            println!("Model Information:");
            println!("  Vocabulary size: {}", training_data.vocab.len());
            println!("  Embedding dimension: {}", model.config.embedding_dim);
            println!("  Model type: {:?}", model.config.model_type);
            println!("  Training epochs: {}", model.config.epochs);
            println!("  Learning rate: {}", model.config.learning_rate);
            println!("  Context window: {}", model.config.context_window);
            
            // Show some sample words and their embeddings
            println!("\nSample words and embeddings:");
            for i in 0..std::cmp::min(10, training_data.reverse_vocab.len()) {
                let word = &training_data.reverse_vocab[i];
                let embedding = model.get_embedding(word, &training_data);
                if let Some(emb) = embedding {
                    let norm = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    println!("  {}: norm={:.3}", word, norm);
                }
            }
        }
        
        Commands::Export { model, vocab: _vocab, output, format } => {
            info!("Exporting embeddings...");
            
            let model_data = fs::read_to_string(&model)
                .unwrap_or_else(|_| {
                    error!("Failed to read model file: {}", model);
                    std::process::exit(1);
                });
            
            let (model, training_data): (EmbeddingModel, TrainingData) = serde_json::from_str(&model_data)
                .unwrap_or_else(|_| {
                    error!("Failed to deserialize model");
                    std::process::exit(1);
                });
            
            match format.as_str() {
                "text" => {
                    if let Err(e) = model.save_embeddings(&output, &training_data) {
                        error!("Failed to export embeddings: {}", e);
                        std::process::exit(1);
                    }
                    info!("Embeddings exported to text format: {}", output);
                }
                "json" => {
                    let export_data: Vec<(String, Vec<f32>)> = training_data.reverse_vocab
                        .iter()
                        .enumerate()
                        .map(|(i, word)| {
                            let embedding = model.get_embedding(word, &training_data)
                                .unwrap_or_else(|| {
                                    error!("Failed to get embedding for word: {}", word);
                                    std::process::exit(1);
                                });
                            (word.clone(), embedding.to_vec())
                        })
                        .collect();
                    
                    let json_data = serde_json::to_string_pretty(&export_data)
                        .unwrap_or_else(|_| {
                            error!("Failed to serialize embeddings to JSON");
                            std::process::exit(1);
                        });
                    
                    fs::write(&output, json_data)
                        .unwrap_or_else(|_| {
                            error!("Failed to write JSON file: {}", output);
                            std::process::exit(1);
                        });
                    
                    info!("Embeddings exported to JSON format: {}", output);
                }
                "bin" => {
                    let bin_data = bincode::serialize(&(&model, &training_data))
                        .unwrap_or_else(|_| {
                            error!("Failed to serialize embeddings to binary");
                            std::process::exit(1);
                        });
                    
                    fs::write(&output, bin_data)
                        .unwrap_or_else(|_| {
                            error!("Failed to write binary file: {}", output);
                            std::process::exit(1);
                        });
                    
                    info!("Embeddings exported to binary format: {}", output);
                }
                _ => {
                    error!("Unknown export format: {}. Use text, json, or bin", format);
                    std::process::exit(1);
                }
            }
        }
    }
}