use embedding::*;

fn main() -> Result<(), String> {
    // Sample training data (or load from file)
    let text = if let Ok(data) = std::fs::read_to_string("examples/data.txt") {
        data
    } else {
        "the quick brown fox jumps over the lazy dog. the fox is quick and the dog is lazy.".to_string()
    };

    // Load and prepare data
    let sentences = load_text_data(&text);
    println!("Loaded {} sentences", sentences.len());
    
    let (vocab, reverse_vocab) = build_vocab(&sentences);
    println!("Built vocabulary with {} words", vocab.len());
    
    // Show some vocabulary
    println!("Sample vocabulary:");
    for (i, word) in reverse_vocab.iter().enumerate().take(10) {
        println!("  {}: {}", i, word);
    }
    
    let training_data = TrainingData {
        sentences,
        vocab,
        reverse_vocab,
    };
    
    // Create training configuration
    let config = TrainingConfig {
        embedding_dim: 10,  // Small dimension for demo
        learning_rate: 0.1, // Higher learning rate for demo
        epochs: 5,
        batch_size: 32,
        context_window: 2,
        negative_samples: 5,
        model_type: ModelType::SkipGram,
        lr_schedule: LearningRateSchedule::Constant,
        early_stopping: None,
        l2_regularization: None,
        gradient_clip: None,
    };
    
    println!("Training with config: {:?}", config);
    
    // Initialize and train model
    let mut model = EmbeddingModel::new(config, training_data.vocab.len());
    
    // Train the model
    model.train(&training_data)?;
    
    // Test similarity calculation
    if let Some(similarity) = model.similarity("fox", "dog", &training_data) {
        println!("Similarity between 'fox' and 'dog': {:.4}", similarity);
    } else {
        println!("Could not calculate similarity (words not in vocabulary)");
    }
    
    if let Some(similarity) = model.similarity("quick", "fox", &training_data) {
        println!("Similarity between 'quick' and 'fox': {:.4}", similarity);
    } else {
        println!("Could not calculate similarity (words not in vocabulary)");
    }
    
    // Save embeddings
    model.save_embeddings("demo_embeddings.txt", &training_data)?;
    println!("Embeddings saved to demo_embeddings.txt");
    
    Ok(())
}