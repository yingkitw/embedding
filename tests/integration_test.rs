use embedding::*;

fn make_test_data() -> TrainingData {
    let text = "the cat sat on the mat. the dog sat on the log. the cat chased the dog.";
    let sentences = load_text_data(text);
    let (vocab, reverse_vocab) = build_vocab(&sentences);
    TrainingData { sentences, vocab, reverse_vocab }
}

fn test_config(model_type: ModelType) -> TrainingConfig {
    TrainingConfig::new(model_type)
        .with_dim(8)
        .with_learning_rate(0.1)
        .with_batch_size(4)
        .with_window(1)
        .with_negative_samples(2)
}

#[test]
fn test_end_to_end_training_pipeline() {
    let data = make_test_data();
    let config = test_config(ModelType::SkipGram).with_epochs(3);

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
    let config = test_config(ModelType::Cbow).with_epochs(2);

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

    let config_sg = test_config(ModelType::SkipGram).with_epochs(3);
    let config_cbow = test_config(ModelType::Cbow).with_epochs(3);

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
        let config = test_config(ModelType::SkipGram)
            .with_epochs(2)
            .with_lr_schedule(schedule);

        let mut model = EmbeddingModel::new(config, data.vocab.len());
        assert!(model.train(&data).is_ok());
        assert!(model.get_embedding("cat", &data).is_some());
    }
}

#[test]
fn test_evaluation_metrics_bounds() {
    let data = make_test_data();
    let config = test_config(ModelType::SkipGram).with_epochs(3);

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
    let config = test_config(ModelType::SkipGram).with_epochs(2);

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
    let mut config = test_config(ModelType::SkipGram)
        .with_epochs(2)
        .with_validation_ratio(0.3);

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
    let config = test_config(ModelType::SkipGram).with_epochs(2);

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
    let config = test_config(ModelType::SkipGram).with_epochs(2);

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
    let config = test_config(ModelType::SkipGram).with_epochs(2);

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
    let config = test_config(ModelType::Cbow).with_epochs(3);

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
            let config = test_config(ModelType::SkipGram).with_epochs(2);
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
            let config = test_config(ModelType::SkipGram).with_epochs(2);
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

#[test]
fn test_transformer_encoder_shapes_and_variance() {
    let encoder = TransformerEncoder::new(2, 2, 8, 16, 10);

    // All-zeros input should produce non-zero output due to position encoding + weights
    let zeros = ndarray::Array2::zeros((4, 8));
    let encoded = encoder.encode_sequence(&zeros);
    assert_eq!(encoded.nrows(), 4);
    assert_eq!(encoded.ncols(), 8);

    // Single token
    let single = ndarray::Array2::zeros((1, 8));
    let encoded_single = encoder.encode_sequence(&single);
    assert_eq!(encoded_single.nrows(), 1);
    assert_eq!(encoded_single.ncols(), 8);

    // Position encoding produces different values for different positions
    let pos_0 = encoder.position_encoding(3);
    assert_eq!(pos_0.nrows(), 3);
    assert_eq!(pos_0.ncols(), 8);

    // Position 0 and position 1 should differ
    let mut diff_found = false;
    for c in 0..8 {
        if (pos_0[[0, c]] - pos_0[[1, c]]).abs() > 1e-6 {
            diff_found = true;
            break;
        }
    }
    assert!(diff_found, "Position encoding should vary across positions");
}

#[test]
fn test_benchmark_evaluator_load_and_correlation() {
    let tsv = "cat\tdog\t0.8\ncat\tmat\t0.2\ndog\tlog\t0.1\nfox\tdog\t0.5\n";
    let pairs = BenchmarkEvaluator::load_from_tsv(tsv);
    assert_eq!(pairs.len(), 4);
    assert_eq!(pairs[0].word1, "cat");
    assert_eq!(pairs[0].word2, "dog");
    assert!((pairs[0].score - 0.8).abs() < 1e-6);

    let data = make_test_data();
    let config = test_config(ModelType::SkipGram).with_epochs(3);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let result = BenchmarkEvaluator::evaluate(&model, &data, &pairs);
    assert_eq!(result.num_pairs, 4);
    assert!(result.num_evaluated <= 4);
    assert!(result.correlation >= -1.0 && result.correlation <= 1.0);
    assert_eq!(result.model_scores.len(), result.num_evaluated);
    assert_eq!(result.human_scores.len(), result.num_evaluated);
}

#[test]
fn test_training_history_json_export() {
    let data = make_test_data();
    let config = test_config(ModelType::SkipGram).with_epochs(3);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    assert!(!model.training_history.epochs.is_empty());
    let json = model.training_history.to_json().unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.get("epochs").is_some());

    let epochs = parsed["epochs"].as_array().unwrap();
    assert!(!epochs.is_empty());
    assert!(epochs[0].get("epoch").is_some());
    assert!(epochs[0].get("loss").is_some());
    assert!(epochs[0].get("learning_rate").is_some());

    let avg_loss = model.training_history.average_loss();
    assert!(avg_loss >= 0.0);
    let final_loss = model.training_history.final_loss();
    assert!(final_loss >= 0.0);
}

#[test]
fn test_incremental_trainer_end_to_end() {
    let mut data = TrainingData::from_text("the cat sat on the mat");
    let config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(4)
        .with_epochs(1)
        .with_batch_size(2)
        .with_window(1)
        .with_negative_samples(1);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let original_vocab = data.vocab.len();
    let new_sentences = vec![
        vec!["newword".to_string(), "cat".to_string()],
    ];

    IncrementalTrainer::update(&mut model, &mut data, &new_sentences, 1).unwrap();

    // Vocabulary should have grown
    assert!(data.vocab.len() > original_vocab);
    // New word should be in vocab
    assert!(data.vocab.contains_key("newword"));
    // Existing word should still be there
    assert!(data.vocab.contains_key("cat"));

    // Model should be able to embed new words
    assert!(model.get_embedding("newword", &data).is_some());
}

#[test]
fn test_incremental_stream_train() {
    let mut data = TrainingData::from_text("hello world foo bar");
    let config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(4)
        .with_epochs(1)
        .with_batch_size(2)
        .with_window(1)
        .with_negative_samples(1);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let stream = vec![
        vec!["stream".to_string(), "word".to_string()],
        vec!["another".to_string(), "stream".to_string()],
    ];

    IncrementalTrainer::stream_train(&mut model, &mut data, stream.into_iter(), 1, 1).unwrap();

    assert!(data.vocab.contains_key("stream"));
    assert!(data.vocab.contains_key("word"));
    assert!(data.vocab.contains_key("another"));
}

#[test]
fn test_multimodal_fusion_all_methods() {
    let fusion = MultimodalFusion::new(4, 4, 4);
    let text = ndarray::Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let aux = ndarray::Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);

    // Concatenation
    let concat = fusion.concatenate(&text, &aux);
    assert_eq!(concat.len(), 8);
    assert_eq!(concat[0], 1.0);
    assert_eq!(concat[4], 0.0);

    // Weighted average
    let avg = fusion.weighted_average(&text, &aux, 0.7).unwrap();
    assert_eq!(avg.len(), 4);
    assert!((avg[0] - 0.7).abs() < 1e-6);
    assert!((avg[1] - 0.3).abs() < 1e-6);

    // Mismatched dims should return None
    let short = ndarray::Array1::from_vec(vec![1.0, 2.0]);
    assert!(fusion.weighted_average(&text, &short, 0.5).is_none());

    // Attention fusion
    let attn = fusion.attention_fusion(&text, &aux).unwrap();
    assert_eq!(attn.len(), 4);

    // Cross-modal similarity - orthogonal vectors should be ~0
    let sim = MultimodalFusion::cross_modal_similarity(&text, &aux);
    assert!(sim.abs() < 1e-5, "Orthogonal vectors should have ~0 similarity, got {}", sim);

    // Same vector should be ~1
    let sim_same = MultimodalFusion::cross_modal_similarity(&text, &text);
    assert!((sim_same - 1.0).abs() < 1e-5, "Same vector should have similarity ~1, got {}", sim_same);
}

#[test]
fn test_kmeans_clustering_basic() {
    let data = make_test_data();
    let config = test_config(ModelType::SkipGram).with_epochs(3);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let clusters = search::KMeansClustering::cluster(&model, &data, 3, 20);
    assert!(!clusters.is_empty());
    assert!(clusters.len() <= 3);

    // All words should be assigned to exactly one cluster
    let total_words: usize = clusters.iter().map(|c| c.len()).sum();
    assert_eq!(total_words, data.vocab.len());

    // Clusters should not overlap
    let mut seen = std::collections::HashSet::new();
    for cluster in &clusters {
        for word in cluster {
            assert!(seen.insert(word.clone()), "Word {} in multiple clusters", word);
        }
    }
}

#[test]
fn test_kmeans_hierarchical_comparison() {
    let data = make_test_data();
    let config = test_config(ModelType::SkipGram).with_epochs(3);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    // Both clustering methods should produce valid clusters
    let kmeans = search::KMeansClustering::cluster(&model, &data, 3, 20);
    let hier = search::HierarchicalClustering::cluster(&model, &data, 3);

    assert!(!kmeans.is_empty());
    assert!(!hier.is_empty());

    let kmeans_total: usize = kmeans.iter().map(|c| c.len()).sum();
    let hier_total: usize = hier.iter().map(|c| c.len()).sum();
    assert_eq!(kmeans_total, data.vocab.len());
    assert_eq!(hier_total, data.vocab.len());
}

#[test]
fn test_cross_validation_skipgram_and_cbow() {
    let data = make_test_data();

    for model_type in [ModelType::SkipGram, ModelType::Cbow] {
        let config = test_config(model_type).with_epochs(2);
        let model = EmbeddingModel::new(config, data.vocab.len());

        let cv = model.cross_validate(&data, 3).unwrap();
        assert_eq!(cv.folds, 3);
        assert_eq!(cv.per_fold_metrics.len(), 3);

        // Averaged metrics should be within valid ranges
        assert!(cv.averaged_metrics.accuracy >= 0.0 && cv.averaged_metrics.accuracy <= 1.0);
        assert!(cv.averaged_metrics.f1_score >= 0.0 && cv.averaged_metrics.f1_score <= 1.0);

        // Per-fold metrics should also be valid
        for metrics in &cv.per_fold_metrics {
            assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        }
    }
}

#[test]
fn test_wordpiece_tokenizer_roundtrip() {
    let corpus = vec![
        "hello".to_string(),
        "world".to_string(),
        "hello".to_string(),
        "world".to_string(),
        "helloworld".to_string(),
    ];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50);
    assert!(tokenizer.vocab_size > 0);

    let encoded = tokenizer.encode("hello world");
    assert!(!encoded.is_empty());

    let decoded = tokenizer.decode(&encoded);
    // Decoding may produce "hello world" or a close approximation
    assert!(!decoded.is_empty());

    // Single word
    let single = tokenizer.encode("hello");
    assert!(!single.is_empty());
    let single_decoded = tokenizer.decode(&single);
    assert_eq!(single_decoded, "hello");
}

#[test]
fn test_cpu_backend_operations() {
    let backend = CpuBackend::new();
    assert_eq!(backend.name(), "cpu");

    let emb = backend.init_embeddings(10, 8);
    assert_eq!(emb.nrows(), 10);
    assert_eq!(emb.ncols(), 8);

    let a = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = ndarray::Array1::from_vec(vec![4.0, 5.0, 6.0]);
    let dot = backend.dot(&a, &b);
    assert!((dot - 32.0).abs() < 1e-5, "Expected 32.0, got {}", dot);

    let mut c = a.clone();
    backend.add_scaled(&mut c, &b, 2.0);
    assert_eq!(c.to_vec(), vec![9.0, 12.0, 15.0]);
}

#[test]
fn test_mmap_embeddings_save_and_load() {
    let data = make_test_data();
    let config = test_config(ModelType::SkipGram).with_epochs(2);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let temp = std::env::temp_dir().join("integration_mmap.bin");
    let path = temp.to_str().unwrap();

    model.save_mmapable_format(path, &data).unwrap();
    let mmap = EmbeddingModel::load_mmap(path).unwrap();

    assert_eq!(mmap.vocab_size(), data.vocab.len());
    assert_eq!(mmap.dim(), model.config.embedding_dim);

    let cat_id = data.vocab["cat"];
    let expected: Vec<f32> = model.embeddings.row(cat_id).to_vec();
    assert_eq!(mmap.get("cat").unwrap(), expected);

    let mut count = 0;
    for (word, emb) in mmap.iter() {
        assert!(!word.is_empty());
        assert_eq!(emb.len(), model.config.embedding_dim);
        count += 1;
    }
    assert_eq!(count, data.vocab.len());

    assert!(mmap.get("nonexistent").is_none());

    std::fs::remove_file(path).ok();
}

#[test]
fn test_pretrained_loader_word2vec_text_auto() {
    let temp = std::env::temp_dir().join("integration_pretrained_w2v.txt");
    let path = temp.to_str().unwrap();

    std::fs::write(
        path,
        "3 4\ncat 0.1 0.2 0.3 0.4\ndog 0.5 0.6 0.7 0.8\nfish 0.9 0.0 0.1 0.2\n",
    )
    .unwrap();

    let emb = PretrainedLoader::auto(path).unwrap();
    assert_eq!(emb.dim(), 4);
    assert_eq!(emb.vocab_size(), 3);
    assert!(emb.contains("cat"));
    assert!(emb.contains("dog"));
    assert!(!emb.contains("elephant"));

    let cat = emb.get("cat").unwrap();
    assert_eq!(cat.len(), 4);
    assert!((cat[0] - 0.1).abs() < 1e-6);

    std::fs::remove_file(path).ok();
}

#[test]
fn test_pretrained_embeddings_similarity_and_most_similar() {
    let mut emb = PretrainedEmbeddings::new(3);
    emb.insert("a".to_string(), vec![1.0, 0.0, 0.0]);
    emb.insert("b".to_string(), vec![0.0, 1.0, 0.0]);
    emb.insert("c".to_string(), vec![1.0, 0.0, 0.0]);

    let sim_ab = emb.similarity("a", "b").unwrap();
    assert!(sim_ab.abs() < 1e-5);

    let sim_ac = emb.similarity("a", "c").unwrap();
    assert!((sim_ac - 1.0).abs() < 1e-5);

    assert!(emb.similarity("a", "missing").is_none());

    let similar = emb.most_similar("a", 1);
    assert_eq!(similar.len(), 1);
    assert_eq!(similar[0].0, "c");
}

#[test]
fn test_pretrained_loader_glove_format_explicit() {
    let temp = std::env::temp_dir().join("integration_pretrained_glove.txt");
    let path = temp.to_str().unwrap();

    std::fs::write(path, "2 3\nhello 0.1 0.2 0.3\nworld 0.4 0.5 0.6\n").unwrap();

    let emb = PretrainedLoader::with_format(path, embedding::pretrained::PretrainedFormat::GloVe)
        .unwrap();
    assert_eq!(emb.dim(), 3);
    assert_eq!(emb.vocab_size(), 2);
    assert_eq!(emb.get("hello").unwrap(), &[0.1, 0.2, 0.3]);

    std::fs::remove_file(path).ok();
}

#[test]
fn test_pretrained_init_model_from_pretrained() {
    let temp = std::env::temp_dir().join("integration_pretrained_init.txt");
    let path = temp.to_str().unwrap();

    std::fs::write(
        path,
        "3 8\ncat 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1\ndog 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2\nthe 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3\n",
    )
    .unwrap();

    let data = make_test_data();
    let config = test_config(ModelType::SkipGram);
    let model = EmbeddingModel::new_with_pretrained(config, data.vocab.len(), &data, path).unwrap();

    let cat_id = data.vocab["cat"];
    let cat_emb = model.embeddings.row(cat_id);
    for &v in cat_emb.iter() {
        assert!((v - 0.1).abs() < 1e-5);
    }

    std::fs::remove_file(path).ok();
}

#[test]
fn test_pretrained_loader_from_mmap_binary() {
    let data = make_test_data();
    let config = test_config(ModelType::SkipGram).with_epochs(2);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data).unwrap();

    let temp = std::env::temp_dir().join("integration_pretrained_mmap.bin");
    let path = temp.to_str().unwrap();

    model.save_mmapable_format(path, &data).unwrap();

    let emb = PretrainedLoader::auto(path).unwrap();
    assert_eq!(emb.dim(), model.config.embedding_dim);
    assert_eq!(emb.vocab_size(), data.vocab.len());

    let cat_id = data.vocab["cat"];
    let expected: Vec<f32> = model.embeddings.row(cat_id).to_vec();
    assert_eq!(emb.get("cat").unwrap(), expected.as_slice());

    std::fs::remove_file(path).ok();
}
