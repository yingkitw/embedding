use embedding::*;

fn main() -> Result<(), String> {
    // Load training data in one step
    let data = if let Ok(text) = std::fs::read_to_string("examples/data.txt") {
        TrainingData::from_text(&text)
    } else {
        TrainingData::from_text(
            "the quick brown fox jumps over the lazy dog. \
             the fox is quick and the dog is lazy. \
             cats chase mice and dogs chase cats."
        )
    };

    println!("Loaded {} sentences, {} vocabulary words", data.sentences.len(), data.vocab.len());

    // Show sample vocabulary
    println!("Sample vocabulary:");
    for (i, word) in data.reverse_vocab.iter().enumerate().take(10) {
        println!("  {}: {}", i, word);
    }

    // Configure training with fluent builder pattern
    let config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(10)
        .with_learning_rate(0.1)
        .with_epochs(5)
        .with_batch_size(32)
        .with_window(2)
        .with_negative_samples(5);

    println!("Training Skip-gram model...");

    // Train model
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data)?;

    // Show training history (learning curve)
    println!("\nTraining History:");
    for epoch in &model.training_history.epochs {
        println!("  Epoch {}: loss={:.4}, lr={:.6}", epoch.epoch, epoch.loss, epoch.learning_rate);
    }

    // Word similarity
    println!("\nWord Similarities:");
    if let Some(sim) = model.similarity("fox", "dog", &data) {
        println!("  fox <-> dog:     {:.4}", sim);
    }
    if let Some(sim) = model.similarity("quick", "fox", &data) {
        println!("  quick <-> fox:   {:.4}", sim);
    }
    if let Some(sim) = model.similarity("cat", "dog", &data) {
        println!("  cat <-> dog:     {:.4}", sim);
    }

    // Word analogy
    println!("\nAnalogy: cat is to dog as mouse is to ?");
    for (word, score) in model.analogy("cat", "dog", "mouse", &data, 3) {
        println!("  {} ({:.4})", word, score);
    }

    // Semantic search
    println!("\nSemantic search for 'cat':");
    for (word, score) in model.semantic_search("cat", &data, 5) {
        println!("  {} ({:.4})", word, score);
    }

    // Cross-validation
    println!("\n5-fold Cross-Validation:");
    let cv = model.cross_validate(&data, 5)?;
    println!("  Averaged Accuracy:  {:.4}", cv.averaged_metrics.accuracy);
    println!("  Averaged F1:        {:.4}", cv.averaged_metrics.f1_score);

    // K-means clustering
    println!("\nK-means Clustering (k=3):");
    let clusters = search::KMeansClustering::cluster(&model, &data, 3, 20);
    for (i, cluster) in clusters.iter().enumerate() {
        println!("  Cluster {}: {:?}", i, cluster);
    }

    // Save embeddings
    model.save_embeddings("demo_embeddings.txt", &data)?;
    println!("\nEmbeddings saved to demo_embeddings.txt");

    // Export training history as JSON
    let history_json = model.training_history.to_json()?;
    std::fs::write("demo_history.json", history_json).map_err(|e| e.to_string())?;
    println!("Training history saved to demo_history.json");

    Ok(())
}