use embedding::pretrained::{PretrainedEmbeddings, PretrainedLoader};
use embedding::*;
use std::fs;

fn main() -> Result<(), String> {
    // --- 1. Train a small model and export in multiple formats ---
    println!("=== Training a small model ===");
    let data = TrainingData::from_text(
        "the cat sat on the mat. the dog sat on the log. \
         the cat chased the dog. the dog chased the cat. \
         fish swim in water. birds fly in sky."
    );
    let config = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(8)
        .with_epochs(3)
        .with_batch_size(4)
        .with_window(2)
        .with_negative_samples(2);
    let mut model = EmbeddingModel::new(config, data.vocab.len());
    model.train(&data)?;
    println!("Trained model: {} vocab, {} dim", data.vocab.len(), model.config.embedding_dim);

    // --- 2. Save as memory-mappable binary format ---
    println!("\n=== Saving to mmapable binary ===");
    let mmap_path = "demo_mmap.bin";
    model.save_mmapable_format(mmap_path, &data)?;
    println!("Saved to {}", mmap_path);

    // --- 3. Load via memory mapping (zero-copy for the file) ---
    println!("\n=== Loading via memory mapping ===");
    let mmap = EmbeddingModel::load_mmap(mmap_path)?;
    println!("Mmap loaded: {} words, {} dim", mmap.vocab_size(), mmap.dim());

    if let Some(emb) = mmap.get("cat") {
        println!("cat embedding (first 3 values): {:.4?}", &emb[..3.min(emb.len())]);
    }

    // Iterate a few words
    println!("\nSample mmap lookups:");
    for (word, emb) in mmap.iter().take(5) {
        println!("  {} -> dim={}, first_val={:.4}", word, emb.len(), emb[0]);
    }

    // --- 4. Save as Word2Vec text format (for pre-trained compatibility) ---
    println!("\n=== Saving as Word2Vec text format ===");
    let w2v_path = "demo_pretrained.txt";
    model.save_word2vec_format(w2v_path, &data)?;
    println!("Saved to {}", w2v_path);

    // --- 5. Load via PretrainedLoader (auto-detects format) ---
    println!("\n=== Loading via PretrainedLoader ===");
    let pretrained = PretrainedLoader::auto(w2v_path)?;
    println!("Pretrained loaded: {} words, {} dim", pretrained.vocab_size(), pretrained.dim());

    // Similarity lookup (no TrainingData needed)
    println!("\nPre-trained similarity lookups:");
    if let Some(sim) = pretrained.similarity("cat", "dog") {
        println!("  cat <-> dog similarity: {:.4}", sim);
    }
    if let Some(sim) = pretrained.similarity("cat", "fish") {
        println!("  cat <-> fish similarity: {:.4}", sim);
    }

    // Top-k most similar
    println!("\nTop 3 words most similar to 'cat':");
    for (word, score) in pretrained.most_similar("cat", 3) {
        println!("  {} ({:.4})", word, score);
    }

    // --- 6. Initialize a new model from pretrained weights ---
    println!("\n=== Training from pretrained initialization ===");
    let config2 = TrainingConfig::new(ModelType::SkipGram)
        .with_dim(8)
        .with_epochs(1);
    let _model2 = EmbeddingModel::new_with_pretrained(
        config2, data.vocab.len(), &data, w2v_path
    )?;
    println!("Initialized model from pretrained file: {} vocab", data.vocab.len());

    // --- 7. PretrainedEmbeddings manual construction (for small dictionaries) ---
    println!("\n=== Manual PretrainedEmbeddings ===");
    let mut manual = PretrainedEmbeddings::new(3);
    manual.insert("king".to_string(),   vec![1.0, 0.0, 0.0]);
    manual.insert("queen".to_string(), vec![0.9, 0.1, 0.0]);
    manual.insert("man".to_string(),    vec![0.0, 1.0, 0.0]);
    manual.insert("woman".to_string(),  vec![0.1, 0.9, 0.0]);

    println!("Manual vocab size: {}", manual.vocab_size());
    println!("king <-> queen similarity: {:.4}", manual.similarity("king", "queen").unwrap());
    println!("Top 2 similar to 'king':");
    for (word, score) in manual.most_similar("king", 2) {
        println!("  {} ({:.4})", word, score);
    }

    // --- Cleanup ---
    fs::remove_file(mmap_path).ok();
    fs::remove_file(w2v_path).ok();
    println!("\nDone!");

    Ok(())
}
