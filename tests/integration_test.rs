use embedding::*;

fn make_test_data() -> TrainingData {
    let text = "the cat sat on the mat. the dog sat on the log. the cat chased the dog.";
    let sentences = load_text_data(text);
    let (vocab, reverse_vocab) = build_vocab(&sentences);
    TrainingData { sentences, vocab, reverse_vocab }
}

#[test]
fn test_end_to_end_training_pipeline() {
    let data = make_test_data();
    let config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 3,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).expect("Training should succeed");

    // Verify embeddings exist for all vocab words
    for word in data.reverse_vocab.iter() {
        assert!(
            model.get_embedding(word, &data).is_some(),
            "Missing embedding for word: {}",
            word
        );
    }

    // Verify similarity computation
    let sim = model.similarity("cat", "dog", &data);
    assert!(sim.is_some(), "Similarity should be computable for known words");

    // Verify analogy solver doesn't panic
    let results = model.analogy("cat", "dog", "log", &data, 3);
    assert!(results.len() <= 3);
}

#[test]
fn test_save_and_load_embeddings() {
    let data = make_test_data();
    let config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 2,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::Cbow,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config.clone(), data.vocab.len());
    model.train(&data).unwrap();

    // Save in Word2Vec format
    let temp_path = std::env::temp_dir().join("integration_word2vec.txt");
    let path_str = temp_path.to_str().unwrap();
    model.save_word2vec_format(path_str, &data).unwrap();

    // Load and verify structure
    let (loaded, dim) = EmbeddingModel::load_word2vec_format(path_str).unwrap();
    assert_eq!(dim, 8);
    assert!(loaded.contains_key("cat"));
    assert!(loaded.contains_key("dog"));
    assert_eq!(loaded.get("cat").unwrap().len(), 8);

    // Save in default format
    let temp_path2 = std::env::temp_dir().join("integration_default.txt");
    let path_str2 = temp_path2.to_str().unwrap();
    model.save_embeddings(path_str2, &data).unwrap();

    let contents = std::fs::read_to_string(path_str2).unwrap();
    assert!(contents.contains("cat"));
    assert!(contents.contains("dog"));

    std::fs::remove_file(path_str).ok();
    std::fs::remove_file(path_str2).ok();
}

#[test]
fn test_text_processing_pipeline() {
    let processor = TextProcessor {
        lowercase: true,
        remove_punctuation: true,
        remove_numbers: true,
        remove_html: true,
        remove_urls: true,
        expand_contractions: true,
        remove_stop_words: false,
        normalize_unicode: false,
        language: "en".to_string(),
    };

    let text = "<p>Visit https://example.com! It's a test with 123 numbers.</p>";
    let sentences = processor.process_text(text);
    assert_eq!(sentences.len(), 1);
    // "visit" stays (not a URL), URL removed, "it's" -> "it is", numbers removed
    assert_eq!(sentences[0], vec!["visit", "it", "is", "a", "test", "with", "numbers"]);
}

#[test]
fn test_cbow_and_skipgram_produce_different_results() {
    let data = make_test_data();

    let config_sg = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 3,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let config_cbow = TrainingConfig {
        model_type: ModelType::Cbow,
        ..config_sg.clone()
    };

    let mut model_sg = EmbeddingModel::new(config_sg, data.vocab.len());
    model_sg.train(&data).unwrap();

    let mut model_cbow = EmbeddingModel::new(config_cbow, data.vocab.len());
    model_cbow.train(&data).unwrap();

    // Both should have embeddings for "cat"
    let emb_sg = model_sg.get_embedding("cat", &data).unwrap();
    let emb_cbow = model_cbow.get_embedding("cat", &data).unwrap();

    // They should not be identical (different training methods)
    let mut identical = true;
    for i in 0..emb_sg.len() {
        if (emb_sg[i] - emb_cbow[i]).abs() > 1e-6 {
            identical = false;
            break;
        }
    }
    assert!(!identical, "SkipGram and CBOW should produce different embeddings");
}

#[test]
fn test_learning_rate_schedule_convergence() {
    let data = make_test_data();

    // Test that different LR schedules all converge (don't panic, produce embeddings)
    let schedules = vec![
        LearningRateSchedule::Constant,
        LearningRateSchedule::Exponential { decay_rate: 0.9 },
        LearningRateSchedule::Step { step_size: 1, gamma: 0.5 },
        LearningRateSchedule::Cosine { t_max: 2 },
    ];

    for schedule in schedules {
        let config = TrainingConfig {
            embedding_dim: 8,
            learning_rate: 0.1,
            epochs: 2,
            batch_size: 4,
            context_window: 1,
            negative_samples: 2,
            model_type: ModelType::SkipGram,
            lr_schedule: schedule,
            early_stopping: None,
            l2_regularization: None,
            gradient_clip: None,
            validation_ratio: None,
        };

        let mut model = EmbeddingModel::new(config, data.vocab.len());
        assert!(model.train(&data).is_ok());
        assert!(model.get_embedding("cat", &data).is_some());
    }
}

#[test]
fn test_evaluation_metrics_bounds() {
    let data = make_test_data();
    let config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 3,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let val_data = model.create_validation_data(&data.sentences);
    let metrics = model.evaluate(&data, &val_data);

    // All metrics should be within valid ranges
    assert!((0.0..=1.0).contains(&metrics.accuracy), "Accuracy out of range: {}", metrics.accuracy);
    assert!((0.0..=1.0).contains(&metrics.precision), "Precision out of range: {}", metrics.precision);
    assert!((0.0..=1.0).contains(&metrics.recall), "Recall out of range: {}", metrics.recall);
    assert!((0.0..=1.0).contains(&metrics.f1_score), "F1 out of range: {}", metrics.f1_score);
    assert!(metrics.mean_similarity >= -1.0 && metrics.mean_similarity <= 1.0);
    assert!((0.0..=1.0).contains(&metrics.embedding_quality_score));
}

#[test]
fn test_evaluate_empty_validation() {
    let data = make_test_data();
    let config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 2,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let empty_val = ValidationData {
        positive_pairs: vec![],
        negative_pairs: vec![],
        analogies: vec![],
    };
    let metrics = model.evaluate(&data, &empty_val);
    assert_eq!(metrics.accuracy, 0.0);
    assert_eq!(metrics.precision, 0.0);
    assert_eq!(metrics.recall, 0.0);
    assert_eq!(metrics.f1_score, 0.0);
    assert_eq!(metrics.mean_similarity, 0.0);
}

#[test]
fn test_train_with_validation_ratio_config() {
    let data = make_test_data();
    let mut config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 2,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: Some(0.3),
    };

    let mut model = EmbeddingModel::new(config.clone(), data.vocab.len());
    model.train(&data).unwrap();

    // Re-split manually and validate
    let (train, val) = model.split_data(&data.sentences, 0.7);
    assert!(!train.is_empty());
    assert!(!val.is_empty());

    let val_data = TrainingData {
        sentences: val,
        vocab: data.vocab.clone(),
        reverse_vocab: data.reverse_vocab.clone(),
    };
    let validation_pairs = model.create_validation_data(&val_data.sentences);
    let metrics = model.evaluate(&val_data, &validation_pairs);
    assert!((0.0..=1.0).contains(&metrics.accuracy));

    // Test with 0.0 validation ratio (no split)
    config.validation_ratio = Some(0.0);
    let mut model2 = EmbeddingModel::new(config, data.vocab.len());
    assert!(model2.train(&data).is_ok());
}

#[test]
fn test_create_validation_data_edge_cases() {
    let data = make_test_data();
    let config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 2,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let model = EmbeddingModel::new(config, data.vocab.len());

    // Single sentence
    let single = vec![vec!["hello".to_string(), "world".to_string()]];
    let val = model.create_validation_data(&single);
    assert_eq!(val.positive_pairs.len(), 1);
    assert!(val.negative_pairs.is_empty());

    // Empty sentences
    let empty: Vec<Vec<String>> = vec![];
    let val_empty = model.create_validation_data(&empty);
    assert!(val_empty.positive_pairs.is_empty());
    assert!(val_empty.negative_pairs.is_empty());
    assert!(val_empty.analogies.is_empty());

    // Two-word sentence only
    let two_word = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
    let val2 = model.create_validation_data(&two_word);
    assert!(!val2.positive_pairs.is_empty());
    assert!(!val2.negative_pairs.is_empty());
}

#[test]
fn test_split_data_produces_correct_sizes() {
    let data = make_test_data();
    let config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 2,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let model = EmbeddingModel::new(config, data.vocab.len());

    let (train, val) = model.split_data(&data.sentences, 0.7);
    assert_eq!(train.len() + val.len(), data.sentences.len());

    let (train2, val2) = model.split_data(&data.sentences, 0.5);
    assert_eq!(train2.len() + val2.len(), data.sentences.len());

    let (train3, val3) = model.split_data(&data.sentences, 1.0);
    assert_eq!(train3.len(), data.sentences.len());
    assert!(val3.is_empty());
}

#[test]
fn test_validation_metrics_json_roundtrip() {
    let data = make_test_data();
    let config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 2,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let val_data = model.create_validation_data(&data.sentences);
    let metrics = model.evaluate(&data, &val_data);

    let json = serde_json::to_string_pretty(&metrics).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.get("accuracy").is_some());
    assert!(parsed.get("precision").is_some());
    assert!(parsed.get("recall").is_some());
    assert!(parsed.get("f1_score").is_some());
    assert!(parsed.get("mean_similarity").is_some());
    assert!(parsed.get("embedding_quality_score").is_some());
}

#[test]
fn test_cbow_validation_metrics() {
    let data = make_test_data();
    let config = TrainingConfig {
        embedding_dim: 8,
        learning_rate: 0.1,
        epochs: 3,
        batch_size: 4,
        context_window: 1,
        negative_samples: 2,
        model_type: ModelType::Cbow,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let val_data = model.create_validation_data(&data.sentences);
    let metrics = model.evaluate(&data, &val_data);
    assert!((0.0..=1.0).contains(&metrics.accuracy));
    assert!((0.0..=1.0).contains(&metrics.f1_score));
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_similarity_range(word1 in "[a-z]{3,8}", word2 in "[a-z]{3,8}") {
            let text = format!("{} {} other words here.", word1, word2);
            let sentences = load_text_data(&text);
            let (vocab, reverse_vocab) = build_vocab(&sentences);
            let data = TrainingData { sentences, vocab, reverse_vocab };
            let config = TrainingConfig {
                embedding_dim: 8,
                learning_rate: 0.1,
                epochs: 2,
                batch_size: 4,
                context_window: 1,
                negative_samples: 2,
                model_type: ModelType::SkipGram,
                lr_schedule: LearningRateSchedule::Constant,
                early_stopping: None,
                l2_regularization: None,
                gradient_clip: None,
                validation_ratio: None,
            };
            let mut model = EmbeddingModel::new(config, data.vocab.len());
            model.train(&data).unwrap();

            if let Some(sim) = model.similarity(&word1, &word2, &data) {
                prop_assert!((-1.0..=1.0).contains(&sim), "Similarity {} out of range [-1, 1]", sim);
            }
        }

        #[test]
        fn prop_normalize_produces_unit_norm(word in "[a-z]{3,8}") {
            let text = format!("{} other words here for context.", word);
            let sentences = load_text_data(&text);
            let (vocab, reverse_vocab) = build_vocab(&sentences);
            let data = TrainingData { sentences, vocab, reverse_vocab };
            let config = TrainingConfig {
                embedding_dim: 8,
                learning_rate: 0.1,
                epochs: 2,
                batch_size: 4,
                context_window: 1,
                negative_samples: 2,
                model_type: ModelType::SkipGram,
                lr_schedule: LearningRateSchedule::Constant,
                early_stopping: None,
                l2_regularization: None,
                gradient_clip: None,
                validation_ratio: None,
            };
            let mut model = EmbeddingModel::new(config, data.vocab.len());
            model.train(&data).unwrap();
            model.normalize_embeddings();

            if let Some(emb) = model.get_embedding(&word, &data) {
                let norm = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    prop_assert!((norm - 1.0).abs() < 1e-5, "Norm {} != 1.0 after normalization", norm);
                }
            }
        }
    }
}
