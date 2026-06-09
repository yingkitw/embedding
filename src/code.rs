/// Processes source code into tokenized sentences for embedding training.
///
/// Strips comments and string literals, tokenizes identifiers/keywords,
/// and optionally splits camelCase / PascalCase names into sub-words.
#[derive(Debug, Clone)]
pub struct CodeProcessor {
    /// Target language hint (e.g. "rust", "python", "javascript").
    /// Affects comment syntax when `remove_comments` is true.
    pub language: String,
    pub remove_comments: bool,
    pub remove_string_literals: bool,
    pub split_camel_case: bool,
    pub remove_numbers: bool,
}

impl Default for CodeProcessor {
    fn default() -> Self {
        Self {
            language: "rust".to_string(),
            remove_comments: true,
            remove_string_literals: true,
            split_camel_case: true,
            remove_numbers: true,
        }
    }
}

impl CodeProcessor {
    /// Converts raw source code into tokenized sentences (one per logical line).
    pub fn process_code(&self, code: &str) -> Vec<Vec<String>> {
        let cleaned = if self.remove_comments {
            self.strip_comments(code)
        } else {
            code.to_string()
        };

        let mut sentences = Vec::new();
        for line in cleaned.lines() {
            let tokens = self.tokenize_line(line);
            if !tokens.is_empty() {
                sentences.push(tokens);
            }
        }
        sentences
    }

    fn strip_comments(&self, code: &str) -> String {
        let mut result = String::with_capacity(code.len());
        let chars: Vec<char> = code.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Multi-line comment /* ... */
            if chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
                i += 2;
                while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                    i += 1;
                }
                i += 2;
                continue;
            }

            // Single-line comment // or #
            if chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            if self.language == "python" && chars[i] == '#' {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }

            // String literals "..." or '...'
            if self.remove_string_literals && (chars[i] == '"' || chars[i] == '\'') {
                let quote = chars[i];
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 2;
                        continue;
                    }
                    if chars[i] == quote {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                continue;
            }

            result.push(chars[i]);
            i += 1;
        }

        result
    }

    fn tokenize_line(&self, line: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in line.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    self.push_token(&mut tokens, &current);
                    current.clear();
                }
            }
        }

        if !current.is_empty() {
            self.push_token(&mut tokens, &current);
        }

        tokens
    }

    fn push_token(&self, out: &mut Vec<String>, raw: &str) {
        if raw.len() > 1 && self.split_camel_case {
            let parts = split_camel_case(raw);
            for part in parts {
                if self.is_valid_token(&part) {
                    out.push(part);
                }
            }
        } else if self.is_valid_token(raw) {
            out.push(raw.to_string());
        }
    }

    fn is_valid_token(&self, token: &str) -> bool {
        if token.is_empty() || token.len() > 64 {
            return false;
        }
        if self.remove_numbers && token.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        true
    }
}

/// Splits a camelCase or PascalCase identifier into sub-words.
///
/// Examples:
/// - `camelCase` → `["camel", "Case"]`
/// - `PascalCase` → `["Pascal", "Case"]`
/// - `HTTPResponse` → `["HTTP", "Response"]`
/// - `snake_case` → `["snake", "case"]` (underscores are already removed)
fn split_camel_case(input: &str) -> Vec<String> {
    if input.len() <= 2 || input.contains('_') {
        return vec![input.to_lowercase()];
    }

    let mut result = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = input.chars().collect();

    for (i, &ch) in chars.iter().enumerate() {
        if ch.is_ascii_uppercase() {
            if !current.is_empty() {
                // Detect acronym boundary: HTTPResponse -> HTTP + Response
                // An uppercase letter preceded by uppercase and followed by
                // lowercase signals the end of an acronym run.
                let prev_is_upper = i > 0 && chars[i - 1].is_ascii_uppercase();
                let next_is_lower = i + 1 < chars.len() && chars[i + 1].is_ascii_lowercase();

                if prev_is_upper && next_is_lower && current.len() > 1 {
                    // Acronym boundary: push the acronym and start a new word.
                    result.push(current.to_lowercase());
                    current.clear();
                } else if !prev_is_upper {
                    // Normal camelCase / PascalCase boundary.
                    result.push(current.to_lowercase());
                    current.clear();
                }
            }
            current.push(ch);
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        result.push(current.to_lowercase());
    }

    result
}

/// Loads source code from a string and processes it with default settings.
pub fn load_code_data(code: &str) -> Vec<Vec<String>> {
    let processor = CodeProcessor::default();
    processor.process_code(code)
}

/// Loads source code from a string using a custom processor.
pub fn load_code_data_advanced(code: &str, processor: &CodeProcessor) -> Vec<Vec<String>> {
    processor.process_code(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_comments_c_style() {
        let processor = CodeProcessor {
            language: "rust".to_string(),
            remove_comments: true,
            remove_string_literals: false,
            split_camel_case: false,
            remove_numbers: true,
        };

        let code = "let x = 5; // initialize\n/* block\ncomment */\nlet y = 10;";
        let sentences = processor.process_code(code);

        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains(&"let".to_string()));
        assert!(!sentences[0].iter().any(|t| t.contains("initialize")));
        assert!(!sentences[1].iter().any(|t| t.contains("comment")));
    }

    #[test]
    fn test_strip_comments_python() {
        let processor = CodeProcessor {
            language: "python".to_string(),
            remove_comments: true,
            remove_string_literals: false,
            split_camel_case: false,
            remove_numbers: true,
        };

        let code = "x = 5  # initialize\ny = 10";
        let sentences = processor.process_code(code);

        assert_eq!(sentences.len(), 2);
        assert!(!sentences[0].iter().any(|t| t.contains("initialize")));
    }

    #[test]
    fn test_remove_string_literals() {
        let processor = CodeProcessor {
            language: "rust".to_string(),
            remove_comments: true,
            remove_string_literals: true,
            split_camel_case: false,
            remove_numbers: true,
        };

        let code = r#"let msg = "hello world"; let x = 1;"#;
        let sentences = processor.process_code(code);

        assert_eq!(sentences.len(), 1);
        assert!(sentences[0].contains(&"let".to_string()));
        assert!(sentences[0].contains(&"msg".to_string()));
        assert!(!sentences[0].iter().any(|t| t.contains("hello")));
    }

    #[test]
    fn test_camel_case_splitting() {
        let processor = CodeProcessor {
            language: "rust".to_string(),
            remove_comments: true,
            remove_string_literals: true,
            split_camel_case: true,
            remove_numbers: true,
        };

        let code = "fn computeEmbeddingVector() {}";
        let sentences = processor.process_code(code);

        assert_eq!(sentences.len(), 1);
        let tokens = &sentences[0];
        assert!(tokens.contains(&"compute".to_string()));
        assert!(tokens.contains(&"embedding".to_string()));
        assert!(tokens.contains(&"vector".to_string()));
    }

    #[test]
    fn test_pascal_case_splitting() {
        let processor = CodeProcessor {
            language: "rust".to_string(),
            remove_comments: true,
            remove_string_literals: true,
            split_camel_case: true,
            remove_numbers: true,
        };

        let code = "struct EmbeddingModel;";
        let sentences = processor.process_code(code);

        assert_eq!(sentences.len(), 1);
        let tokens = &sentences[0];
        assert!(tokens.contains(&"embedding".to_string()));
        assert!(tokens.contains(&"model".to_string()));
    }

    #[test]
    fn test_acronym_preservation() {
        assert_eq!(
            split_camel_case("HTTPResponse"),
            vec!["http", "response"]
        );
        assert_eq!(
            split_camel_case("URLLoader"),
            vec!["url", "loader"]
        );
    }

    #[test]
    fn test_no_camel_split() {
        let processor = CodeProcessor {
            language: "rust".to_string(),
            remove_comments: true,
            remove_string_literals: true,
            split_camel_case: false,
            remove_numbers: true,
        };

        let code = "fn computeEmbeddingVector() {}";
        let sentences = processor.process_code(code);

        let tokens = &sentences[0];
        assert!(tokens.contains(&"computeEmbeddingVector".to_string()));
        assert!(!tokens.contains(&"compute".to_string()));
    }

    #[test]
    fn test_load_code_data() {
        let code = "let x = 5;\nlet y = 10;";
        let sentences = load_code_data(code);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], vec!["let", "x"]);
        assert_eq!(sentences[1], vec!["let", "y"]);
    }

    #[test]
    fn test_rust_code_processing() {
        let code = r#"
            // This is a comment
            fn main() {
                let vector = vec![1, 2, 3];
                println!("hello");
            }
        "#;
        let sentences = load_code_data(code);
        assert!(!sentences.is_empty());

        // Comments and string literals should be gone
        for sentence in &sentences {
            for token in sentence {
                assert_ne!(token, "hello");
                assert_ne!(token, "comment");
            }
        }

        // Identifiers should be present
        let all_tokens: Vec<String> = sentences.into_iter().flatten().collect();
        assert!(all_tokens.contains(&"fn".to_string()));
        assert!(all_tokens.contains(&"main".to_string()));
        assert!(all_tokens.contains(&"let".to_string()));
        assert!(all_tokens.contains(&"vector".to_string()));
    }
}
