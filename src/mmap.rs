use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

/// Magic bytes for the mmapable embedding format.
const MMAP_MAGIC: &[u8] = b"EMBD";
/// Current format version.
const MMAP_VERSION: u32 = 1;

/// Read-only memory-mapped embedding storage.
///
/// This struct maps a binary embedding file directly into process memory,
/// allowing access to very large embedding files without loading them into
/// the Rust heap. Lookups are performed by word string via an in-memory
/// index of offsets into the mmap.
///
/// # Example
/// ```no_run
/// use embedding::MmapEmbeddings;
/// let mmap = MmapEmbeddings::open("embeddings.bin").unwrap();
/// if let Some(emb) = mmap.get("cat") {
///     println!("cat embedding dim: {}", emb.len());
/// }
/// ```
pub struct MmapEmbeddings {
    mmap: Mmap,
    vocab: HashMap<String, (usize, usize)>, // word -> (offset in data section, len)
    dim: usize,
    data_offset: usize, // byte offset where the flat f32 data begins
}

impl MmapEmbeddings {
    /// Opens a memory-mapped embedding file.
    pub fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("Failed to mmap: {}", e))? };

        if mmap.len() < 20 {
            return Err("File too small for header".to_string());
        }

        // Header: magic (4) + version (4) + vocab_size (8) + dim (8) = 24 bytes
        if &mmap[0..4] != MMAP_MAGIC {
            return Err("Invalid magic bytes".to_string());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != MMAP_VERSION {
            return Err(format!("Unsupported version: {}", version));
        }
        let vocab_size = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let dim = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;

        // Vocab index section
        let mut offset = 24usize;
        let mut vocab = HashMap::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            if offset + 8 > mmap.len() {
                return Err("Corrupt vocab index".to_string());
            }
            let word_len = u64::from_le_bytes(
                mmap[offset..offset + 8].try_into().unwrap()
            ) as usize;
            offset += 8;

            if offset + word_len > mmap.len() {
                return Err("Corrupt vocab entry".to_string());
            }
            let word = String::from_utf8_lossy(&mmap[offset..offset + word_len]).to_string();
            offset += word_len;

            if offset + 16 > mmap.len() {
                return Err("Corrupt vocab data offset".to_string());
            }
            let data_offset = u64::from_le_bytes(
                mmap[offset..offset + 8].try_into().unwrap()
            ) as usize;
            let data_len = u64::from_le_bytes(
                mmap[offset + 8..offset + 16].try_into().unwrap()
            ) as usize;
            offset += 16;

            vocab.insert(word, (data_offset, data_len));
        }

        Ok(Self {
            mmap,
            vocab,
            dim,
            data_offset: offset,
        })
    }

    /// Looks up an embedding by word. Returns `None` if the word is not in vocabulary.
    pub fn get(&self, word: &str) -> Option<Vec<f32>> {
        let &(data_offset, data_len) = self.vocab.get(word)?;
        let byte_offset = self.data_offset + data_offset;
        let byte_end = byte_offset + data_len;
        if byte_end > self.mmap.len() {
            return None;
        }
        let bytes = &self.mmap[byte_offset..byte_end];
        // Copy to aligned Vec to avoid UB from unaligned pointer dereference
        let mut floats = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            floats.push(f32::from_ne_bytes(arr));
        }
        Some(floats)
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Iterates over all (word, embedding) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Vec<f32>)> + '_ {
        self.vocab.iter().map(move |(word, &(off, len))| {
            let byte_offset = self.data_offset + off;
            let byte_end = byte_offset + len;
            let bytes = &self.mmap[byte_offset..byte_end];
            let mut floats = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                floats.push(f32::from_ne_bytes(arr));
            }
            (word.as_str(), floats)
        })
    }
}

/// Saves embeddings to a memory-mappable binary file.
///
/// Format:
/// - Header (24 bytes): magic "EMBD" + version (u32 LE) + vocab_size (u64 LE) + dim (u64 LE)
/// - Vocab index: repeated [word_len (u64 LE), word_bytes, data_offset (u64 LE), data_len (u64 LE)]
/// - Data section: flat native-endian f32 values
pub fn save_mmapable_format(
    path: &str,
    words: &[String],
    embeddings: &[Vec<f32>],
) -> Result<(), String> {
    if words.len() != embeddings.len() {
        return Err("Words and embeddings length mismatch".to_string());
    }
    let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
    for emb in embeddings {
        if emb.len() != dim {
            return Err("Inconsistent embedding dimensions".to_string());
        }
    }

    let mut file = File::create(path).map_err(|e| e.to_string())?;

    // Header
    file.write_all(MMAP_MAGIC).map_err(|e| e.to_string())?;
    file.write_all(&MMAP_VERSION.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&(words.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&(dim as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;

    // Vocab index (we write placeholders and backfill later)
    let index_start = file.metadata().map_err(|e| e.to_string())?.len() as usize;
    let mut index_entries: Vec<(/*word_len*/u64, /*word*/Vec<u8>, /*data_offset*/u64, /*data_len*/u64)> =
        Vec::with_capacity(words.len());

    // Write vocab index
    for word in words {
        let word_bytes = word.as_bytes();
        file.write_all(&(word_bytes.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(word_bytes).map_err(|e| e.to_string())?;
        // Placeholder for offset and len
        file.write_all(&0u64.to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(&0u64.to_le_bytes()).map_err(|e| e.to_string())?;
    }

    // Data section
    let data_start = file.metadata().map_err(|e| e.to_string())?.len() as usize;
    for emb in embeddings {
        for &val in emb {
            file.write_all(&val.to_ne_bytes()).map_err(|e| e.to_string())?;
        }
    }

    // Backfill offsets and lengths
    let mut current_offset = 0u64;
    for (i, emb) in embeddings.iter().enumerate() {
        let data_len = (emb.len() * 4) as u64;
        index_entries.push((
            words[i].len() as u64,
            words[i].as_bytes().to_vec(),
            current_offset,
            data_len,
        ));
        current_offset += data_len;
    }

    // Actually we need to backfill in the file. Let me rewrite the approach:
    // It's simpler to build the file in-memory and write once.
    drop(file);
    std::fs::remove_file(path).ok();

    let mut buf: Vec<u8> = Vec::new();
    // Header
    buf.extend_from_slice(MMAP_MAGIC);
    buf.extend_from_slice(&MMAP_VERSION.to_le_bytes());
    buf.extend_from_slice(&(words.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(dim as u64).to_le_bytes());

    // Build index and data
    let mut index = Vec::new();
    let mut data = Vec::new();
    for (word, emb) in words.iter().zip(embeddings.iter()) {
        let word_bytes = word.as_bytes();
        let offset = data.len() as u64;
        let len = (emb.len() * 4) as u64;
        index.extend_from_slice(&(word_bytes.len() as u64).to_le_bytes());
        index.extend_from_slice(word_bytes);
        index.extend_from_slice(&offset.to_le_bytes());
        index.extend_from_slice(&len.to_le_bytes());
        for &val in emb {
            data.extend_from_slice(&val.to_ne_bytes());
        }
    }

    buf.extend_from_slice(&index);
    buf.extend_from_slice(&data);

    std::fs::write(path, buf).map_err(|e| e.to_string())?;
    Ok(())
}
