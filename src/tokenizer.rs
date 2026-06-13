use std::collections::HashMap;

/// Byte Pair Encoding tokenizer with trained merge rules.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct BPETokenizer {
    pub vocab: HashMap<String, usize>,
    pub merges: Vec<(String, String)>,
    pub vocab_size: usize,
}


impl BPETokenizer {
    /// Trains BPE merge rules from a corpus until the target vocabulary size is reached.
    pub fn train(corpus: &[String], target_vocab_size: usize) -> Self {
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();

        // Initialize: split words into characters + end-of-word marker
        for word in corpus {
            let cleaned = word.to_lowercase().trim().to_string();
            if cleaned.is_empty() {
                continue;
            }
            let chars: Vec<String> = cleaned.chars().map(|c| c.to_string()).collect();
            let mut tokens = chars;
            tokens.push("</w>".to_string());
            *word_freqs.entry(tokens).or_insert(0) += 1;
        }

        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut merges: Vec<(String, String)> = Vec::new();

        // Build initial vocabulary from all characters
        let mut all_tokens: Vec<String> = Vec::new();
        for tokens in word_freqs.keys() {
            for token in tokens {
                all_tokens.push(token.clone());
            }
        }
        for token in all_tokens {
            let next_id = vocab.len();
            vocab.entry(token).or_insert(next_id);
        }

        while vocab.len() < target_vocab_size {
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();

            for (tokens, freq) in &word_freqs {
                for pair in tokens.windows(2) {
                    let key = (pair[0].clone(), pair[1].clone());
                    *pair_counts.entry(key).or_insert(0) += freq;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find most frequent pair
            let best_pair = pair_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(pair, _)| pair)
                .unwrap();

            let merged = format!("{}{}", best_pair.0, best_pair.1);
            let next_id = vocab.len();
            vocab.insert(merged.clone(), next_id);
            merges.push(best_pair.clone());

            // Apply merge to all word representations
            let mut new_word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
            for (tokens, freq) in word_freqs {
                let mut new_tokens = Vec::new();
                let mut i = 0;
                while i < tokens.len() {
                    if i + 1 < tokens.len()
                        && tokens[i] == best_pair.0
                        && tokens[i + 1] == best_pair.1
                    {
                        new_tokens.push(merged.clone());
                        i += 2;
                    } else {
                        new_tokens.push(tokens[i].clone());
                        i += 1;
                    }
                }
                *new_word_freqs.entry(new_tokens).or_insert(0) += freq;
            }
            word_freqs = new_word_freqs;
        }

        let vocab_size = vocab.len();
        Self {
            vocab,
            merges,
            vocab_size,
        }
    }

    /// Tokenizes a word using the learned BPE merge rules.
    pub fn encode(&self, text: &str) -> Vec<String> {
        let word = text.to_lowercase();
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        tokens.push("</w>".to_string());

        for (a, b) in &self.merges {
            let merged = format!("{}{}", a, b);
            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && &tokens[i] == a && &tokens[i + 1] == b {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        tokens
    }

    /// Reconstructs the original word from BPE tokens.
    pub fn decode(&self, tokens: &[String]) -> String {
        let text = tokens.join("");
        text.replace("</w>", " ").trim().to_string()
    }
}

/// WordPiece tokenizer with a fixed vocabulary and greedy longest-match encoding.
///
/// Used by BERT-style models. Subword pieces that are not word beginnings
/// are prefixed with `##`.
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    pub vocab: HashMap<String, usize>,
    pub vocab_size: usize,
    pub unk_token: String,
}

impl WordPieceTokenizer {
    /// Creates a WordPiece tokenizer from an existing vocabulary map.
    pub fn from_vocab(vocab: HashMap<String, usize>) -> Self {
        let vocab_size = vocab.len();
        Self {
            vocab,
            vocab_size,
            unk_token: "[UNK]".to_string(),
        }
    }

    /// Trains a WordPiece vocabulary from a corpus by iteratively adding
    /// the most frequent subword pairs.
    pub fn train(corpus: &[String], target_vocab_size: usize) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        vocab.insert("[MASK]".to_string(), 4);

        // Start with characters
        let mut char_freqs: HashMap<String, usize> = HashMap::new();
        for word in corpus {
            for ch in word.chars() {
                *char_freqs.entry(ch.to_string()).or_insert(0) += 1;
            }
        }
        for (ch, _) in char_freqs {
            let next_id = vocab.len();
            vocab.entry(ch).or_insert(next_id);
        }

        // Simple heuristic: collect common substrings up to target size
        let mut subword_freqs: HashMap<String, usize> = HashMap::new();
        for word in corpus {
            let w = word.to_lowercase();
            for len in 2..=w.len().min(8) {
                for i in 0..=w.len() - len {
                    let sub = &w[i..i + len];
                    *subword_freqs.entry(sub.to_string()).or_insert(0) += 1;
                }
            }
        }

        let mut sorted: Vec<(String, usize)> = subword_freqs.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (sub, _) in sorted {
            if vocab.len() >= target_vocab_size {
                break;
            }
            let next_id = vocab.len();
            vocab.entry(sub).or_insert(next_id);
        }

        let vocab_size = vocab.len();
        Self {
            vocab,
            vocab_size,
            unk_token: "[UNK]".to_string(),
        }
    }

    /// Encodes a word into WordPiece tokens using greedy longest-match.
    pub fn encode_word(&self, word: &str) -> Vec<String> {
        let text = word.to_lowercase();
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut longest_match: Option<String> = None;
            let mut longest_len = 0;

            for j in (i + 1)..=chars.len() {
                let candidate: String = chars[i..j].iter().collect();
                if self.vocab.contains_key(&candidate) && candidate.len() > longest_len {
                    longest_match = Some(candidate);
                    longest_len = j - i;
                }
            }

            if let Some(token) = longest_match {
                let prefix = if i == 0 { "" } else { "##" };
                tokens.push(format!("{}{}", prefix, token));
                i += longest_len;
            } else {
                tokens.push(self.unk_token.clone());
                i += 1;
            }
        }

        tokens
    }

    /// Encodes a full sentence into WordPiece tokens.
    pub fn encode(&self, text: &str) -> Vec<String> {
        let mut all_tokens = Vec::new();
        for word in text.split_whitespace() {
            all_tokens.extend(self.encode_word(word));
        }
        all_tokens
    }

    /// Decodes WordPiece tokens back into a string.
    pub fn decode(&self, tokens: &[String]) -> String {
        let mut words = Vec::new();
        let mut current_word = String::new();

        for token in tokens {
            if let Some(suffix) = token.strip_prefix("##") {
                current_word.push_str(suffix);
            } else if !current_word.is_empty() {
                words.push(current_word.clone());
                current_word = token.clone();
            } else {
                current_word = token.clone();
            }
        }
        if !current_word.is_empty() {
            words.push(current_word);
        }

        words.join(" ")
    }
}
