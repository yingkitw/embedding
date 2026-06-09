use clap::{Parser, Subcommand};
use crate::commands::*;

#[derive(Parser)]
#[command(name = "embedding-train")]
#[command(about = "A CLI tool for training word embeddings from scratch")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
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

        /// Model type: skipgram or cbow
        #[arg(short, long, default_value = "skipgram")]
        model_type: String,

        /// Treat input as source code instead of natural language text
        #[arg(long)]
        code: bool,

        /// Programming language for code preprocessing (rust, python, javascript, etc.)
        #[arg(long, default_value = "rust")]
        language: String,
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

        /// Export format: text, json, bin, or word2vec
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Train a model and enter interactive mode for queries
    Interactive {
        /// Input text file for training
        #[arg(short, long)]
        input: String,

        /// Output model file path
        #[arg(short, long, default_value = "model.json")]
        output: String,

        /// Embedding dimension
        #[arg(short, long, default_value = "100")]
        dim: usize,

        /// Number of training epochs
        #[arg(short, long, default_value = "10")]
        epochs: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.025")]
        learning_rate: f64,

        /// Context window size
        #[arg(short, long, default_value = "5")]
        window: usize,

        /// Number of negative samples
        #[arg(short, long, default_value = "5")]
        negative_samples: usize,

        /// Model type: skipgram or cbow
        #[arg(short, long, default_value = "skipgram")]
        model: String,
    },
}

pub fn run(cli: Cli) {
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
            code,
            language,
        } => handle_train(input, output, embeddings, config_path, dim, learning_rate, epochs, batch_size, window, negative_samples, model_type, code, language),
        Commands::Similarity { word1, word2, model, vocab: _vocab } => {
            handle_similarity(word1, word2, model);
        }
        Commands::Info { model, vocab: _vocab } => {
            handle_info(model);
        }
        Commands::Export { model, vocab: _vocab, output, format } => {
            handle_export(model, output, format);
        }
        Commands::Interactive { input, output, dim, epochs, learning_rate, window, negative_samples, model } => {
            handle_interactive(input, output, dim, epochs, learning_rate, window, negative_samples, model);
        }
    }
}

