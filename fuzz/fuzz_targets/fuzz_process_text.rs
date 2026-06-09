#![no_main]

use libfuzzer_sys::fuzz_target;
use embedding::TextProcessor;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let processor = TextProcessor {
            lowercase: true,
            remove_punctuation: true,
            remove_numbers: false,
            remove_html: true,
            remove_urls: true,
            expand_contractions: true,
            remove_stop_words: false,
            language: "en".to_string(),
        };
        // Should not panic on any input
        let _ = processor.process_text(text);
    }
});
