use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

/// Builds a vocabulary map and reverse lookup from tokenized sentences.
pub fn build_vocab(sentences: &[Vec<String>]) -> (HashMap<String, usize>, Vec<String>) {
    let mut vocab = HashMap::new();
    let mut reverse_vocab = Vec::new();
    let mut vocab_counter = 0;
    
    for sentence in sentences {
        for word in sentence {
            if !vocab.contains_key(word) {
                vocab.insert(word.clone(), vocab_counter);
                reverse_vocab.push(word.clone());
                vocab_counter += 1;
            }
        }
    }
    
    (vocab, reverse_vocab)
}

/// Configurable text preprocessing pipeline.
///
/// Controls lowercasing, punctuation removal, HTML stripping,
/// URL removal, contraction expansion, and stop-word filtering.
#[derive(Debug, Clone)]
pub struct TextProcessor {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub remove_numbers: bool,
    pub remove_stop_words: bool,
    pub remove_html: bool,
    pub remove_urls: bool,
    pub expand_contractions: bool,
    pub normalize_unicode: bool,
    pub language: String,
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            remove_numbers: false,
            remove_stop_words: false,
            remove_html: false,
            remove_urls: false,
            expand_contractions: false,
            normalize_unicode: false,
            language: "en".to_string(),
        }
    }
}

impl TextProcessor {
    /// Processes raw text into tokenized sentences according to the configured filters.
    pub fn process_text(&self, text: &str) -> Vec<Vec<String>> {
        let mut text = text.nfc().collect::<String>();

        // Remove HTML tags
        if self.remove_html {
            text = Self::strip_html(&text);
        }

        // Remove URLs
        if self.remove_urls {
            text = Self::strip_urls(&text);
        }

        let mut sentences = Vec::new();

        // Split into sentences
        for sentence in text.split(['.', '!', '?', '\n']) {
            if !sentence.trim().is_empty() {
                let mut processed_words = Vec::new();

                // Split into words and process each word
                for word in sentence.split_whitespace() {
                    let processed_word = self.process_word(word);
                    if !processed_word.is_empty() {
                        for subword in processed_word.split_whitespace() {
                            processed_words.push(subword.to_string());
                        }
                    }
                }

                if !processed_words.is_empty() {
                    sentences.push(processed_words);
                }
            }
        }

        sentences
    }

    fn strip_html(text: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;
        for c in text.chars() {
            if c == '<' {
                in_tag = true;
            } else if c == '>' {
                in_tag = false;
            } else if !in_tag {
                result.push(c);
            }
        }
        result
    }

    fn strip_urls(text: &str) -> String {
        text.split_whitespace()
            .filter(|word| !(word.starts_with("http://") || word.starts_with("https://") || word.starts_with("www.")))
            .collect::<Vec<&str>>()
            .join(" ")
    }

    fn process_word(&self, word: &str) -> String {
        let mut result = word.to_string();

        // Expand contractions
        if self.expand_contractions {
            result = Self::expand_contraction(&result);
        }

        // Convert to lowercase
        if self.lowercase {
            result = result.to_lowercase();
        }

        // Remove punctuation
        if self.remove_punctuation {
            result = result.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect::<String>()
                .trim()
                .to_string();
        }

        // Remove numbers
        if self.remove_numbers {
            result = result.chars()
                .filter(|c| !c.is_ascii_digit())
                .collect::<String>();
        }

        // Remove empty strings
        if result.is_empty() {
            return String::new();
        }

        result
    }

    fn expand_contraction(word: &str) -> String {
        match word.to_lowercase().as_str() {
            "can't" => "cannot".to_string(),
            "won't" => "will not".to_string(),
            "n't" => " not".to_string(),
            "'re" => " are".to_string(),
            "'ve" => " have".to_string(),
            "'ll" => " will".to_string(),
            "'d" => " would".to_string(),
            "'m" => " am".to_string(),
            "i'm" => "i am".to_string(),
            "don't" => "do not".to_string(),
            "doesn't" => "does not".to_string(),
            "didn't" => "did not".to_string(),
            "isn't" => "is not".to_string(),
            "aren't" => "are not".to_string(),
            "wasn't" => "was not".to_string(),
            "weren't" => "were not".to_string(),
            "haven't" => "have not".to_string(),
            "hasn't" => "has not".to_string(),
            "hadn't" => "had not".to_string(),
            "wouldn't" => "would not".to_string(),
            "couldn't" => "could not".to_string(),
            "shouldn't" => "should not".to_string(),
            "let's" => "let us".to_string(),
            "that's" => "that is".to_string(),
            "who's" => "who is".to_string(),
            "what's" => "what is".to_string(),
            "here's" => "here is".to_string(),
            "there's" => "there is".to_string(),
            "where's" => "where is".to_string(),
            "it's" => "it is".to_string(),
            _ => word.to_string(),
        }
    }
    
    /// Simple heuristic-based language detection.
    pub fn detect_language(&self, text: &str) -> String {
        // Simple heuristic for language detection
        // This is a very basic implementation - in practice, you'd use more sophisticated methods
        
        let english_stop_words = ["the", "and", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"];
        
        let words_vec: Vec<&str> = text.split_whitespace().collect();
        let words = &words_vec;
        let mut english_count = 0;
        
        for word in words {
            let lower_word = word.to_lowercase();
            if english_stop_words.contains(&lower_word.as_str()) {
                english_count += 1;
            }
        }
        
        // If more than 20% of words are common English stop words, assume English
        if english_count > words.len() / 5 {
            "en".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

/// Tokenizes text using the default [`TextProcessor`] settings.
pub fn load_text_data(text: &str) -> Vec<Vec<String>> {
    let processor = TextProcessor::default();
    processor.process_text(text)
}

/// Tokenizes text using a custom [`TextProcessor`].
pub fn load_text_data_advanced(text: &str, processor: &TextProcessor) -> Vec<Vec<String>> {
    processor.process_text(text)
}
