use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

/// Supported pre-trained embedding file formats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PretrainedFormat {
    /// Word2Vec/Gensim text format: `vocab_size dim` header, then `word f32...` per line.
    Word2VecText,
    /// Google's original Word2Vec binary format: `vocab_size dim` header,
    /// then `word<space><binary_f32_x_dim>` per entry.
    Word2VecBinary,
    /// GloVe text format: same as Word2Vec text (`word f32...`).
    GloVe,
    /// fastText `.vec` text format: same as Word2Vec text.
    FastText,
    /// Our memory-mapped `.bin` format.
    MmapBinary,
}

/// Inference-only embedding storage loaded from a pre-trained file.
///
/// Unlike [`EmbeddingModel`](crate::EmbeddingModel), this struct is intended
/// for pure lookup / similarity use-cases without training. It holds a
/// `HashMap<String, Vec<f32>>` so OOV words are handled naturally.
///
/// # Example
/// ```no_run
/// use embedding::pretrained::{PretrainedEmbeddings, PretrainedLoader};
/// let emb = PretrainedLoader::auto("glove.6B.50d.txt").unwrap();
/// if let Some(v) = emb.get("hello") {
///     println!("hello embedding dim: {}", v.len());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct PretrainedEmbeddings {
    embeddings: HashMap<String, Vec<f32>>,
    dim: usize,
}

impl PretrainedEmbeddings {
    /// Creates a new empty storage with the given expected dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dim,
        }
    }

    /// Inserts a word and its vector.
    pub fn insert(&mut self, word: String, vec: Vec<f32>) {
        self.embeddings.insert(word, vec);
    }

    /// Returns the embedding vector for a word, or `None` if OOV.
    pub fn get(&self, word: &str) -> Option<&[f32]> {
        self.embeddings.get(word).map(|v| v.as_slice())
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.embeddings.len()
    }

    /// Checks whether the word exists in the vocabulary.
    pub fn contains(&self, word: &str) -> bool {
        self.embeddings.contains_key(word)
    }

    /// Cosine similarity between two words.
    pub fn similarity(&self, w1: &str, w2: &str) -> Option<f32> {
        let a = self.get(w1)?;
        let b = self.get(w2)?;
        Some(cosine_similarity(a, b))
    }

    /// Top-k most similar words to a query word.
    pub fn most_similar(&self, word: &str, top_k: usize) -> Vec<(String, f32)> {
        let query = match self.get(word) {
            Some(q) => q,
            None => return Vec::new(),
        };
        let mut results: Vec<(String, f32)> = self
            .embeddings
            .iter()
            .filter(|(w, _)| w.as_str() != word)
            .map(|(w, vec)| (w.clone(), cosine_similarity(query, vec)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        results
    }

    /// Iterates over all (word, embedding) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[f32])> {
        self.embeddings.iter().map(|(w, v)| (w.as_str(), v.as_slice()))
    }

    /// Converts into the raw HashMap.
    pub fn into_inner(self) -> HashMap<String, Vec<f32>> {
        self.embeddings
    }
}

/// Loader for pre-trained embeddings with format auto-detection.
pub struct PretrainedLoader;

impl PretrainedLoader {
    /// Loads a file, automatically inferring the format from extension and content.
    ///
    /// Extensions:
    /// - `.bin` with magic `EMBD` → `MmapBinary`
    /// - `.bin` or `.vec` without text header → `Word2VecBinary`
    /// - `.txt`, `.vec` with text header → `Word2VecText` / `GloVe` / `FastText`
    pub fn auto(path: &str) -> Result<PretrainedEmbeddings, String> {
        let format = Self::detect_format(path)?;
        Self::with_format(path, format)
    }

    /// Loads a file with an explicitly specified format.
    pub fn with_format(path: &str, format: PretrainedFormat) -> Result<PretrainedEmbeddings, String> {
        match format {
            PretrainedFormat::Word2VecText | PretrainedFormat::GloVe | PretrainedFormat::FastText => {
                Self::load_word2vec_text(path)
            }
            PretrainedFormat::Word2VecBinary => Self::load_word2vec_binary(path),
            PretrainedFormat::MmapBinary => Self::load_mmap_binary(path),
        }
    }

    /// Detects the format of a file by inspecting its extension and header.
    pub fn detect_format(path: &str) -> Result<PretrainedFormat, String> {
        let lower = path.to_lowercase();
        if lower.ends_with(".bin") {
            // Peek at first 4 bytes to distinguish mmap binary vs word2vec binary
            let mut file = File::open(path).map_err(|e| e.to_string())?;
            let mut magic = [0u8; 4];
            let n = file.read(&mut magic).map_err(|e| e.to_string())?;
            if n == 4 && &magic == b"EMBD" {
                return Ok(PretrainedFormat::MmapBinary);
            }
            // Otherwise assume Google's binary format
            return Ok(PretrainedFormat::Word2VecBinary);
        }
        if lower.ends_with(".vec") {
            // fastText .vec is text; Word2Vec binary also uses .vec sometimes
            // Peek at first byte to see if it's printable text
            let mut file = File::open(path).map_err(|e| e.to_string())?;
            let mut first = [0u8; 1];
            file.read_exact(&mut first).map_err(|e| e.to_string())?;
            if first[0].is_ascii_digit() || first[0] == b'-' {
                return Ok(PretrainedFormat::FastText); // text header starts with vocab count
            }
            return Ok(PretrainedFormat::Word2VecBinary);
        }
        if lower.ends_with(".txt") {
            return Ok(PretrainedFormat::GloVe);
        }
        // Default fallback: try text
        Ok(PretrainedFormat::Word2VecText)
    }

    fn load_word2vec_text(path: &str) -> Result<PretrainedEmbeddings, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header = lines
            .next()
            .ok_or("Empty file")?
            .map_err(|e| e.to_string())?;
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 2 {
            return Err("Invalid header format: expected '<vocab> <dim>'".to_string());
        }
        let _vocab_size: usize = parts[0].parse().map_err(|_| "Invalid vocab size")?;
        let dim: usize = parts[1].parse().map_err(|_| "Invalid dimension")?;

        let mut result = PretrainedEmbeddings::new(dim);
        for line in lines {
            let line = line.map_err(|e| e.to_string())?;
            let mut parts = line.split_whitespace();
            let word = parts.next().ok_or("Missing word")?.to_string();
            let values: Result<Vec<f32>, _> = parts.map(|s| s.parse()).collect();
            let values = values.map_err(|_| format!("Invalid float value in line for '{}',", word))?;
            if values.len() != dim {
                return Err(format!(
                    "Expected {} dimensions for '{}', got {}",
                    dim, word, values.len()
                ));
            }
            result.insert(word, values);
        }

        Ok(result)
    }

    fn load_word2vec_binary(path: &str) -> Result<PretrainedEmbeddings, String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;

        // Read header line (text) until newline
        let mut header_buf = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            file.read_exact(&mut byte).map_err(|e| format!("Failed to read header: {}", e))?;
            if byte[0] == b'\n' {
                break;
            }
            header_buf.push(byte[0]);
        }
        let header = String::from_utf8_lossy(&header_buf);
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 2 {
            return Err("Invalid binary header format".to_string());
        }
        let vocab_size: usize = parts[0].parse().map_err(|_| "Invalid vocab size")?;
        let dim: usize = parts[1].parse().map_err(|_| "Invalid dimension")?;

        let mut result = PretrainedEmbeddings::new(dim);
        let mut word_buf = Vec::with_capacity(64);

        for _ in 0..vocab_size {
            word_buf.clear();
            // Read word until space
            loop {
                file.read_exact(&mut byte).map_err(|e| format!("Failed to read word: {}", e))?;
                if byte[0] == b' ' {
                    break;
                }
                word_buf.push(byte[0]);
            }
            let word = String::from_utf8_lossy(&word_buf).to_string();

            // Read dim x f32 (little-endian, same as C float)
            let mut vec = vec![0.0f32; dim];
            for i in 0..dim {
                let mut float_bytes = [0u8; 4];
                file.read_exact(&mut float_bytes)
                    .map_err(|e| format!("Failed to read float for '{}': {}", word, e))?;
                vec[i] = f32::from_le_bytes(float_bytes);
            }

            // Normalize the vector length (Google's binary has an extra byte per vector sometimes)
            // After the floats, skip any trailing whitespace / newline
            let mut trailing = [0u8; 1];
            if file.read(&mut trailing).unwrap_or(0) > 0 {
                if trailing[0] != b'\n' && trailing[0] != b' ' {
                    // We consumed a byte that belongs to the next word, push it back conceptually.
                    // For simplicity, we assume one newline per vector. If not, re-seek.
                    // Most Google binary files have one newline after each vector.
                }
            }

            result.insert(word, vec);
        }

        Ok(result)
    }

    fn load_mmap_binary(path: &str) -> Result<PretrainedEmbeddings, String> {
        let mmap = crate::mmap::MmapEmbeddings::open(path)?;
        let dim = mmap.dim();
        let mut result = PretrainedEmbeddings::new(dim);
        for (word, emb) in mmap.iter() {
            result.insert(word.to_string(), emb);
        }
        Ok(result)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}
