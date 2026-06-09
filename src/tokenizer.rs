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
