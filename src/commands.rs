use crate::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::SliceRandom;
use std::fs;
use tracing::{info, error};

pub fn handle_train(
    input: String,
    output: String,
    embeddings: String,
    config_path: Option<String>,
    dim: usize,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    window: usize,
    negative_samples: usize,
    model_type: String,
    validation_ratio: f64,
    validation_output: Option<String>,
    is_code: bool,
    language: String,
) {
    info!("Starting embedding training...");

    let text = fs::read_to_string(&input)
        .unwrap_or_else(|_| {
            error!("Failed to read input file: {}", input);
            std::process::exit(1);
        });

    let sentences = if is_code {
        info!("Processing input as {} source code", language);
        let processor = CodeProcessor {
            language,
            ..CodeProcessor::default()
        };
        load_code_data_advanced(&text, &processor)
    } else {
        load_text_data(&text)
    };
    info!("Loaded {} sentences", sentences.len());

    let (vocab, reverse_vocab) = build_vocab(&sentences);
    info!("Built vocabulary with {} words", vocab.len());

    let model_type = match model_type.as_str() {
        "skipgram" => ModelType::SkipGram,
        "cbow" => ModelType::Cbow,
        _ => {
            error!("Unknown model type: {}. Use skipgram or cbow", model_type);
            std::process::exit(1);
        }
    };

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
            gradient_clip: None,
            validation_ratio: None,
        }
    };

    let validation_ratio = config.validation_ratio.unwrap_or(validation_ratio);

    let (train_sentences, val_sentences) = if validation_ratio > 0.0 && !sentences.is_empty() {
        let total = sentences.len();
        let train_size = (total as f64 * (1.0 - validation_ratio)) as usize;
        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(&mut rand::thread_rng());
        let train: Vec<Vec<String>> = indices[..train_size].iter().map(|&i| sentences[i].clone()).collect();
        let val: Vec<Vec<String>> = indices[train_size..].iter().map(|&i| sentences[i].clone()).collect();
        info!("Split data: {} train, {} validation sentences", train.len(), val.len());
        (train, val)
    } else {
        (sentences.clone(), Vec::new())
    };

    let training_data = TrainingData {
        sentences: train_sentences,
        vocab: vocab.clone(),
        reverse_vocab: reverse_vocab.clone(),
    };

    let mut model = EmbeddingModel::new(config.clone(), vocab.len());
    info!("Training model with config: {:?}", config);

    let pb = ProgressBar::new(config.epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} epochs {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    pb.set_message("training...");

    if let Err(e) = model.train(&training_data) {
        pb.finish_with_message("training failed");
        error!("Training failed: {}", e);
        std::process::exit(1);
    }

    pb.finish_with_message("training complete");

    let full_training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };

    let model_data = serde_json::to_string(&(&model, &full_training_data))
        .unwrap_or_else(|_| {
            error!("Failed to serialize model");
            std::process::exit(1);
        });

    fs::write(&output, model_data)
        .unwrap_or_else(|_| {
            error!("Failed to save model to: {}", output);
            std::process::exit(1);
        });

    if let Err(e) = model.save_embeddings(&embeddings, &full_training_data) {
        error!("Failed to save embeddings: {}", e);
        std::process::exit(1);
    }

    if !val_sentences.is_empty() {
        let val_data = TrainingData {
            sentences: val_sentences,
            vocab: full_training_data.vocab.clone(),
            reverse_vocab: full_training_data.reverse_vocab.clone(),
        };
        let validation_pairs = model.create_validation_data(&val_data.sentences);
        let metrics = model.evaluate(&val_data, &validation_pairs);
        info!("Validation metrics: {:?}", metrics);
        println!("\nValidation Results:");
        println!("  Accuracy:  {:.4}", metrics.accuracy);
        println!("  Precision: {:.4}", metrics.precision);
        println!("  Recall:    {:.4}", metrics.recall);
        println!("  F1 Score:  {:.4}", metrics.f1_score);
        println!("  Mean Sim:  {:.4}", metrics.mean_similarity);
        println!("  Quality:   {:.4}", metrics.embedding_quality_score);

        if let Some(path) = validation_output {
            let metrics_json = serde_json::to_string_pretty(&metrics).unwrap_or_else(|_| {
                error!("Failed to serialize validation metrics");
                std::process::exit(1);
            });
            fs::write(&path, metrics_json).unwrap_or_else(|_| {
                error!("Failed to write validation metrics to: {}", path);
                std::process::exit(1);
            });
            info!("Validation metrics saved to: {}", path);
        }
    }

    info!("Training completed successfully!");
    info!("Model saved to: {}", output);
    info!("Embeddings saved to: {}", embeddings);
}

pub fn handle_similarity(word1: String, word2: String, model_path: String) {
    info!("Calculating similarity between '{}' and '{}'", word1, word2);

    let model_data = fs::read_to_string(&model_path)
        .unwrap_or_else(|_| {
            error!("Failed to read model file: {}", model_path);
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

pub fn handle_info(model_path: String) {
    info!("Inspecting model...");

    let model_data = fs::read_to_string(&model_path)
        .unwrap_or_else(|_| {
            error!("Failed to read model file: {}", model_path);
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

pub fn handle_export(model_path: String, output: String, format: String) {
    info!("Exporting embeddings...");

    let model_data = fs::read_to_string(&model_path)
        .unwrap_or_else(|_| {
            error!("Failed to read model file: {}", model_path);
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
                .map(|word| {
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
        "word2vec" => {
            if let Err(e) = model.save_word2vec_format(&output, &training_data) {
                error!("Failed to export embeddings: {}", e);
                std::process::exit(1);
            }
            info!("Embeddings exported to Word2Vec format: {}", output);
        }
        _ => {
            error!("Unknown export format: {}. Use text, json, bin, or word2vec", format);
            std::process::exit(1);
        }
    }
}

pub fn handle_validate(
    model_path: String,
    input: String,
    output: Option<String>,
) {
    info!("Validating model: {}", model_path);

    let model_data = fs::read_to_string(&model_path)
        .unwrap_or_else(|_| {
            error!("Failed to read model file: {}", model_path);
            std::process::exit(1);
        });

    let (model, training_data): (EmbeddingModel, TrainingData) = serde_json::from_str(&model_data)
        .unwrap_or_else(|_| {
            error!("Failed to deserialize model");
            std::process::exit(1);
        });

    let text = fs::read_to_string(&input)
        .unwrap_or_else(|_| {
            error!("Failed to read validation input file: {}", input);
            std::process::exit(1);
        });

    let sentences = load_text_data(&text);
    info!("Loaded {} validation sentences", sentences.len());

    let mut val_vocab = training_data.vocab.clone();
    let mut val_reverse = training_data.reverse_vocab.clone();
    for sentence in &sentences {
        for word in sentence {
            if !val_vocab.contains_key(word.as_str()) {
                let id = val_vocab.len();
                val_vocab.insert(word.clone(), id);
                val_reverse.push(word.clone());
            }
        }
    }

    let val_data = TrainingData {
        sentences,
        vocab: val_vocab,
        reverse_vocab: val_reverse,
    };

    let validation_pairs = model.create_validation_data(&val_data.sentences);
    let metrics = model.evaluate(&val_data, &validation_pairs);
    info!("Validation metrics: {:?}", metrics);

    println!("\nValidation Results:");
    println!("  Accuracy:  {:.4}", metrics.accuracy);
    println!("  Precision: {:.4}", metrics.precision);
    println!("  Recall:    {:.4}", metrics.recall);
    println!("  F1 Score:  {:.4}", metrics.f1_score);
    println!("  Mean Sim:  {:.4}", metrics.mean_similarity);
    println!("  Quality:   {:.4}", metrics.embedding_quality_score);

    if let Some(path) = output {
        let metrics_json = serde_json::to_string_pretty(&metrics).unwrap_or_else(|_| {
            error!("Failed to serialize validation metrics");
            std::process::exit(1);
        });
        fs::write(&path, metrics_json).unwrap_or_else(|_| {
            error!("Failed to write validation metrics to: {}", path);
            std::process::exit(1);
        });
        info!("Validation metrics saved to: {}", path);
    }
}

pub fn handle_interactive(
    input: String,
    output: String,
    dim: usize,
    epochs: usize,
    learning_rate: f64,
    window: usize,
    negative_samples: usize,
    model_type_str: String,
) {
    info!("Interactive training mode");

    let text = fs::read_to_string(&input)
        .unwrap_or_else(|_| {
            error!("Failed to read input file: {}", input);
            std::process::exit(1);
        });

    let sentences = load_text_data(&text);
    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData { sentences, vocab, reverse_vocab };

    let model_type = match model_type_str.as_str() {
        "skipgram" => ModelType::SkipGram,
        "cbow" => ModelType::Cbow,
        _ => {
            error!("Unknown model type: {}. Use skipgram or cbow", model_type_str);
            std::process::exit(1);
        }
    };

    let config = TrainingConfig {
        embedding_dim: dim,
        learning_rate,
        epochs,
        batch_size: 32,
        context_window: window,
        negative_samples,
        model_type,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, training_data.vocab.len());
    info!("Training model...");
    if let Err(e) = model.train(&training_data) {
        error!("Training failed: {}", e);
        std::process::exit(1);
    }

    let model_data = serde_json::to_string(&(&model, &training_data))
        .unwrap_or_else(|_| {
            error!("Failed to serialize model");
            std::process::exit(1);
        });
    fs::write(&output, model_data)
        .unwrap_or_else(|_| {
            error!("Failed to write model file: {}", output);
            std::process::exit(1);
        });
    info!("Model saved to: {}", output);

    println!("\n=== Interactive Mode ===");
    println!("Commands:");
    println!("  sim <word1> <word2>  - Compute similarity");
    println!("  analogy <a> <b> <c>  - Solve analogy a:b :: c:?");
    println!("  search <word>        - Find similar words");
    println!("  quit                 - Exit\n");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    use std::io::Write;

    loop {
        print!("> ");
        stdout.flush().unwrap();
        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() {
            break;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "quit" | "exit" => break,
            "sim" => {
                if parts.len() >= 3 {
                    if let Some(sim) = model.similarity(parts[1], parts[2], &training_data) {
                        println!("Similarity: {:.4}", sim);
                    } else {
                        println!("One or both words not found");
                    }
                } else {
                    println!("Usage: sim <word1> <word2>");
                }
            }
            "analogy" => {
                if parts.len() >= 4 {
                    let results = model.analogy(parts[1], parts[2], parts[3], &training_data, 5);
                    if results.is_empty() {
                        println!("No results found");
                    } else {
                        println!("Top results:");
                        for (word, score) in results {
                            println!("  {}: {:.4}", word, score);
                        }
                    }
                } else {
                    println!("Usage: analogy <word1> <word2> <word3>");
                }
            }
            "search" => {
                if parts.len() >= 2 {
                    let results = model.semantic_search(parts[1], &training_data, 10);
                    if results.is_empty() {
                        println!("No results found");
                    } else {
                        println!("Similar words:");
                        for (word, score) in results {
                            println!("  {}: {:.4}", word, score);
                        }
                    }
                } else {
                    println!("Usage: search <word>");
                }
            }
            _ => println!("Unknown command. Type 'quit' to exit."),
        }
    }
}
