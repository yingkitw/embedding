use embedding::*;

fn main() -> Result<(), String> {
    let text = "the quick brown fox jumps over the lazy dog. \
                the fox is quick and the dog is lazy. \
                cats chase mice and dogs chase cats. \
                the cat sat on the mat. the dog sat on the log. \
                the cat chased the dog. the dog chased the cat. \
                quick brown foxes jump over lazy dogs. \
                cats sleep on mats and dogs sleep on logs. \
                the quick cat chased the lazy dog. \
                mice run from cats and cats run from dogs.";

    let data = TrainingData::from_text(text);
    println!("Loaded {} sentences, {} vocabulary words", data.sentences.len(), data.vocab.len());

    // Demonstrate all new training enhancements
    let config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(16)
        .with_learning_rate(0.05)
        .with_epochs(6)
        .with_batch_size(16)
        .with_window(2)
        .with_negative_samples(5)
        // New: unigram^0.75 negative sampling
        .with_unigram_negative_sampling(true)
        // New: sub-sample frequent words (the, and, on)
        .with_subsample_threshold(Some(1e-3))
        // New: linear LR warm-up for first 2 epochs
        .with_warmup_epochs(Some(2))
        // New: save checkpoint every 2 epochs
        .with_checkpoint_interval(Some(2))
        .with_checkpoint_path(Some("./checkpoints".to_string()))
        // New: parallel training over CPU cores
        .with_parallel(true);

    println!("\nTraining with enhancements:");
    println!("  - Unigram^0.75 negative sampling: enabled");
    println!("  - Sub-sampling threshold: 1e-3");
    println!("  - LR warm-up epochs: 2");
    println!("  - Checkpoint interval: every 2 epochs");
    println!("  - Parallel training: enabled");

    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data)?;

    println!("\nTraining History:");
    for epoch in &model.training_history.epochs {
        println!("  Epoch {}: loss={:.4}, lr={:.6}", epoch.epoch, epoch.loss, epoch.learning_rate);
    }

    // Show word similarities
    println!("\nWord Similarities:");
    if let Some(sim) = model.similarity("cat", "dog", &data) {
        println!("  cat <-> dog:     {:.4}", sim);
    }
    if let Some(sim) = model.similarity("quick", "lazy", &data) {
        println!("  quick <-> lazy:  {:.4}", sim);
    }
    if let Some(sim) = model.similarity("fox", "dog", &data) {
        println!("  fox <-> dog:     {:.4}", sim);
    }

    // Demonstrate checkpoint save / load
    let checkpoint_path = "checkpoints/demo_checkpoint.json";
    model.save_checkpoint(checkpoint_path, 6, 1.0)?;
    println!("\nCheckpoint saved to {}", checkpoint_path);

    let loaded = EmbeddingModel::load_checkpoint(checkpoint_path)?;
    println!("Checkpoint loaded. Vocab size: {}", loaded.vocab_size);

    // Compare sequential vs parallel training
    println!("\n--- Comparing Sequential vs Parallel ---");
    let seq_config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(16)
        .with_learning_rate(0.05)
        .with_epochs(3)
        .with_batch_size(16)
        .with_window(2)
        .with_negative_samples(5)
        .with_parallel(false);

    let par_config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(16)
        .with_learning_rate(0.05)
        .with_epochs(3)
        .with_batch_size(16)
        .with_window(2)
        .with_negative_samples(5)
        .with_parallel(true);

    let start = std::time::Instant::now();
    let mut seq_model = EmbeddingModel::new(seq_config, data.vocab.len());
    seq_model.train(&data)?;
    let seq_time = start.elapsed();

    let start = std::time::Instant::now();
    let mut par_model = EmbeddingModel::new(par_config, data.vocab.len());
    par_model.train(&data)?;
    let par_time = start.elapsed();

    println!("Sequential training: {:?}", seq_time);
    println!("Parallel training:   {:?}", par_time);
    println!("Speed-up:            {:.2}x", seq_time.as_secs_f64() / par_time.as_secs_f64().max(0.001));

    Ok(())
}
