use criterion::{black_box, criterion_group, criterion_main, Criterion};
use embedding::*;

fn bench_skipgram_training(c: &mut Criterion) {
    let sentences = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["jumps".to_string(), "over".to_string(), "the".to_string(), "lazy".to_string()],
        vec!["dog".to_string(), "and".to_string(), "cat".to_string()],
        vec!["machine".to_string(), "learning".to_string(), "is".to_string(), "fun".to_string()],
    ];
    
    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };
    
    let config = TrainingConfig {
        embedding_dim: 100,
        learning_rate: 0.025,
        epochs: 5,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };
    
    c.bench_function("skipgram_training", |b| {
        b.iter(|| {
            let mut model = EmbeddingModel::new(config.clone(), training_data.vocab.len());
            model.train(&training_data).unwrap();
            black_box(());
        })
    });
}

fn bench_cbow_training(c: &mut Criterion) {
    let sentences = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["jumps".to_string(), "over".to_string(), "the".to_string(), "lazy".to_string()],
        vec!["dog".to_string(), "and".to_string(), "cat".to_string()],
        vec!["machine".to_string(), "learning".to_string(), "is".to_string(), "fun".to_string()],
    ];
    
    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };
    
    let config = TrainingConfig {
        embedding_dim: 100,
        learning_rate: 0.025,
        epochs: 5,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::Cbow,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };
    
    c.bench_function("cbow_training", |b| {
        b.iter(|| {
            let mut model = EmbeddingModel::new(config.clone(), training_data.vocab.len());
            model.train(&training_data).unwrap();
            black_box(());
        })
    });
}

fn bench_similarity_calculation(c: &mut Criterion) {
    let sentences = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["jumps".to_string(), "over".to_string(), "the".to_string(), "lazy".to_string()],
        vec!["dog".to_string(), "and".to_string(), "cat".to_string()],
        vec!["machine".to_string(), "learning".to_string(), "is".to_string(), "fun".to_string()],
    ];
    
    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };
    
    let config = TrainingConfig {
        embedding_dim: 100,
        learning_rate: 0.025,
        epochs: 10,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };
    
    let mut model = EmbeddingModel::new(config, training_data.vocab.len());
    model.train(&training_data).unwrap();
    
    c.bench_function("similarity_calculation", |b| {
        b.iter(|| {
            black_box(model.similarity("fox", "dog", &training_data));
        })
    });
}

fn bench_embedding_retrieval(c: &mut Criterion) {
    let sentences = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["jumps".to_string(), "over".to_string(), "the".to_string(), "lazy".to_string()],
        vec!["dog".to_string(), "and".to_string(), "cat".to_string()],
        vec!["machine".to_string(), "learning".to_string(), "is".to_string(), "fun".to_string()],
    ];
    
    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };
    
    let config = TrainingConfig {
        embedding_dim: 100,
        learning_rate: 0.025,
        epochs: 10,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };
    
    let mut model = EmbeddingModel::new(config, training_data.vocab.len());
    model.train(&training_data).unwrap();
    
    let words = vec!["the", "fox", "dog", "machine", "learning", "quick", "brown"];
    
    c.bench_function("embedding_retrieval", |b| {
        b.iter(|| {
            for word in black_box(&words) {
                black_box(model.get_embedding(word, &training_data));
            }
        })
    });
}

fn bench_vocab_building(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog. The fox is quick and the dog is lazy. Machine learning is fascinating and powerful.";
    let sentences = load_text_data(text);
    
    c.bench_function("vocab_building", |b| {
        b.iter(|| {
            black_box(build_vocab(&sentences));
        })
    });
}

fn bench_semantic_search(c: &mut Criterion) {
    let sentences = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["jumps".to_string(), "over".to_string(), "the".to_string(), "lazy".to_string()],
        vec!["dog".to_string(), "and".to_string(), "cat".to_string()],
        vec!["machine".to_string(), "learning".to_string(), "is".to_string(), "fun".to_string()],
    ];

    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };

    let config = TrainingConfig {
        embedding_dim: 100,
        learning_rate: 0.025,
        epochs: 10,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, training_data.vocab.len());
    model.train(&training_data).unwrap();

    c.bench_function("semantic_search", |b| {
        b.iter(|| {
            black_box(model.semantic_search("fox", &training_data, 5));
        })
    });
}

fn bench_analogy(c: &mut Criterion) {
    let sentences = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["jumps".to_string(), "over".to_string(), "the".to_string(), "lazy".to_string()],
        vec!["dog".to_string(), "and".to_string(), "cat".to_string()],
        vec!["machine".to_string(), "learning".to_string(), "is".to_string(), "fun".to_string()],
    ];

    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };

    let config = TrainingConfig {
        embedding_dim: 100,
        learning_rate: 0.025,
        epochs: 10,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, training_data.vocab.len());
    model.train(&training_data).unwrap();

    c.bench_function("analogy", |b| {
        b.iter(|| {
            black_box(model.analogy("fox", "dog", "cat", &training_data, 3));
        })
    });
}

fn bench_lsh_query(c: &mut Criterion) {
    let sentences = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["jumps".to_string(), "over".to_string(), "the".to_string(), "lazy".to_string()],
        vec!["dog".to_string(), "and".to_string(), "cat".to_string()],
        vec!["machine".to_string(), "learning".to_string(), "is".to_string(), "fun".to_string()],
    ];

    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };

    let config = TrainingConfig {
        embedding_dim: 100,
        learning_rate: 0.025,
        epochs: 10,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, training_data.vocab.len());
    model.train(&training_data).unwrap();

    let mut lsh = LSHIndex::new(4, 100);
    lsh.build(&model, &training_data);

    c.bench_function("lsh_query", |b| {
        b.iter(|| {
            black_box(lsh.query("fox", &model, &training_data, 5));
        })
    });
}

fn bench_sentence_embedding(c: &mut Criterion) {
    let sentences = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["jumps".to_string(), "over".to_string(), "the".to_string(), "lazy".to_string()],
        vec!["dog".to_string(), "and".to_string(), "cat".to_string()],
        vec!["machine".to_string(), "learning".to_string(), "is".to_string(), "fun".to_string()],
    ];

    let (vocab, reverse_vocab) = build_vocab(&sentences);
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };

    let config = TrainingConfig {
        embedding_dim: 100,
        learning_rate: 0.025,
        epochs: 10,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
        validation_ratio: None,
    };

    let mut model = EmbeddingModel::new(config, training_data.vocab.len());
    model.train(&training_data).unwrap();

    let sentence = vec!["the".to_string(), "quick".to_string(), "fox".to_string()];

    c.bench_function("sentence_embedding", |b| {
        b.iter(|| {
            black_box(model.sentence_embedding(&sentence, &training_data));
        })
    });
}

criterion_group!(
    benches,
    bench_skipgram_training,
    bench_cbow_training,
    bench_similarity_calculation,
    bench_embedding_retrieval,
    bench_vocab_building,
    bench_semantic_search,
    bench_analogy,
    bench_lsh_query,
    bench_sentence_embedding
);
criterion_main!(benches);