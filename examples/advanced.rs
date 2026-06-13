use embedding::*;
use ndarray::Array1;

fn main() -> Result<(), String> {
    // --- Transformer Encoder Example ---
    println!("=== Transformer Encoder ===");
    let encoder = TransformerEncoder::new(2, 4, 16, 32, 20);
    let tokens = ndarray::Array2::zeros((5, 16));
    let encoded = encoder.encode_sequence(&tokens);
    println!("Input shape:  (5, 16)");
    println!("Output shape: ({}, {})", encoded.nrows(), encoded.ncols());

    // --- Multimodal Fusion Example ---
    println!("\n=== Multimodal Fusion ===");
    let fusion = MultimodalFusion::new(4, 4, 4);
    let text = Array1::from_vec(vec![1.0, 0.5, 0.2, 0.0]);
    let image = Array1::from_vec(vec![0.2, 0.8, 0.1, 0.5]);

    let concat = fusion.concatenate(&text, &image);
    println!("Concatenated dim: {}", concat.len());

    let avg = fusion.weighted_average(&text, &image, 0.6).unwrap();
    println!("Weighted average: {:?}", avg.to_vec());

    let attn = fusion.attention_fusion(&text, &image).unwrap();
    println!("Attention fusion: {:?}", attn.to_vec());

    let cross_sim = MultimodalFusion::cross_modal_similarity(&text, &image);
    println!("Cross-modal similarity: {:.4}", cross_sim);

    // --- Benchmark Evaluation Example ---
    println!("\n=== Benchmark Evaluation ===");
    let data = TrainingData::from_text(
        "the cat sat on the mat. the dog sat on the log. the cat chased the dog."
    );
    let config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(8)
        .with_epochs(3);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data)?;

    let tsv = "cat\tdog\t0.7\ncat\tmat\t0.3\ndog\tlog\t0.5\n";
    let pairs = BenchmarkEvaluator::load_from_tsv(tsv);
    let result = BenchmarkEvaluator::evaluate(&model, &data, &pairs);
    println!("Pairs evaluated: {}/{}", result.num_evaluated, result.num_pairs);
    println!("Spearman correlation: {:.4}", result.correlation);

    // --- Incremental Training Example ---
    println!("\n=== Incremental Training ===");
    let mut data = TrainingData::from_text("hello world");
    let config = TrainingConfig::new(ModelType::SkipGram).with_dim(8).with_epochs(2);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data)?;
    println!("Initial vocab size: {}", data.vocab.len());

    let new_sentences = vec![
        vec!["new".to_string(), "word".to_string(), "here".to_string()],
    ];
    IncrementalTrainer::update(&mut model, &mut data, &new_sentences, 1)?;
    println!("Updated vocab size: {}", data.vocab.len());
    println!("'new' in vocabulary: {}", data.vocab.contains_key("new"));

    // Stream training
    let stream = vec![
        vec!["stream".to_string(), "data".to_string()],
        vec!["more".to_string(), "stream".to_string()],
    ];
    IncrementalTrainer::stream_train(&mut model, &mut data, stream.into_iter(), 1, 1)?;
    println!("After stream training vocab size: {}", data.vocab.len());

    // --- Backend Example ---
    println!("\n=== Compute Backend ===");
    let backend = CpuBackend::new();
    println!("Backend: {}", backend.name());
    let emb = backend.init_embeddings(100, 32);
    println!("Initialized embeddings: {} x {}", emb.nrows(), emb.ncols());

    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
    println!("Dot product: {:.2}", backend.dot(&a, &b));

    Ok(())
}
