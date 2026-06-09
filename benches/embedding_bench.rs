use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use embedding_trainer::*;
use std::time::Duration;

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
    };
    
    c.bench_function("skipgram_training", |b| {
        b.iter(|| {
            let mut model = EmbeddingModel::new(config.clone(), training_data.vocab.len());
            black_box(model.train(&training_data).unwrap());
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
    };
    
    c.bench_function("cbow_training", |b| {
        b.iter(|| {
            let mut model = EmbeddingModel::new(config.clone(), training_data.vocab.len());
            black_box(model.train(&training_data).unwrap());
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

criterion_group!(
    benches,
    bench_skipgram_training,
    bench_cbow_training,
    bench_similarity_calculation,
    bench_embedding_retrieval,
    bench_vocab_building
);
criterion_main!(benches);