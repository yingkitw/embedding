use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use prost::Message;
use crate::{EmbeddingModel, TrainingData};
use crate::onnx::*;
use crate::mmap;

impl EmbeddingModel {
    /// Saves embeddings as a JSON file.
    pub fn save_embeddings(&self, path: &str, data: &TrainingData) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| e.to_string())?;

        for (word_id, word) in data.reverse_vocab.iter().enumerate() {
            let embedding = self.embeddings.row(word_id);
            let embedding_str = embedding.iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");

            writeln!(file, "{}\t{}", word, embedding_str).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Saves embeddings in the Word2Vec/Gensim text format.
    pub fn save_word2vec_format(&self, path: &str, data: &TrainingData) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        let vocab_size = data.reverse_vocab.len();
        let dim = self.config.embedding_dim;

        writeln!(file, "{} {}", vocab_size, dim).map_err(|e| e.to_string())?;

        for (word_id, word) in data.reverse_vocab.iter().enumerate() {
            let embedding = self.embeddings.row(word_id);
            let values: Vec<String> = embedding.iter().map(|&x| format!("{:.6}", x)).collect();
            writeln!(file, "{} {}", word, values.join(" ")).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Loads embeddings from a Word2Vec/Gensim text format file.
    pub fn load_word2vec_format(path: &str) -> Result<(std::collections::HashMap<String, Vec<f32>>, usize), String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header = lines.next()
            .ok_or("Empty file")?
            .map_err(|e| e.to_string())?;
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 2 {
            return Err("Invalid header format".to_string());
        }
        let _vocab_size: usize = parts[0].parse().map_err(|_| "Invalid vocab size")?;
        let dim: usize = parts[1].parse().map_err(|_| "Invalid dimension")?;

        let mut embeddings = std::collections::HashMap::new();
        for line in lines {
            let line = line.map_err(|e| e.to_string())?;
            let mut parts = line.split_whitespace();
            let word = parts.next().ok_or("Missing word")?.to_string();
            let values: Result<Vec<f32>, _> = parts.map(|s| s.parse()).collect();
            let values = values.map_err(|_| "Invalid float value")?;
            if values.len() != dim {
                return Err(format!("Expected {} dimensions, got {}", dim, values.len()));
            }
            embeddings.insert(word, values);
        }

        Ok((embeddings, dim))
    }

    /// Saves embeddings as a NumPy `.npy` file for TensorFlow/PyTorch compatibility.
    pub fn save_numpy_format(&self, path: &str, data: &TrainingData) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        let rows = data.reverse_vocab.len();
        let cols = self.config.embedding_dim;

        // NumPy .npy format header (simplified version 1.0)
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
            rows, cols
        );
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len();
        let padding = (64 - (header_len + 10) % 64) % 64;

        file.write_all(b"\x93NUMPY\x01\x00").map_err(|e| e.to_string())?;
        let total_len = (header_len + padding) as u16;
        file.write_all(&total_len.to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(header_bytes).map_err(|e| e.to_string())?;
        for _ in 0..padding {
            file.write_all(b" ").map_err(|e| e.to_string())?;
        }

        for (word_id, _) in data.reverse_vocab.iter().enumerate() {
            let embedding = self.embeddings.row(word_id);
            for &val in embedding.iter() {
                file.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }

        Ok(())
    }

    /// Saves embeddings as an ONNX model with a Gather node for lookup.
    pub fn save_onnx_format(&self, path: &str, data: &TrainingData) -> Result<(), String> {
        let vocab_size = data.reverse_vocab.len() as i64;
        let dim = self.config.embedding_dim as i64;

        // Flatten embeddings into raw little-endian f32 bytes
        let mut raw_data = Vec::with_capacity((vocab_size as usize) * (dim as usize) * 4);
        for (word_id, _) in data.reverse_vocab.iter().enumerate() {
            let row = self.embeddings.row(word_id);
            for &val in row.iter() {
                raw_data.extend_from_slice(&val.to_le_bytes());
            }
        }

        let embedding_tensor = TensorProto {
            dims: vec![vocab_size, dim],
            data_type: TensorProtoDataType::Float as i32,
            raw_data,
            name: "embeddings".to_string(),
        };

        let gather_node = NodeProto {
            input: vec!["embeddings".to_string(), "input_indices".to_string()],
            output: vec!["output".to_string()],
            name: "gather_embeddings".to_string(),
            op_type: "Gather".to_string(),
            domain: "".to_string(),
        };

        let input_type = TypeProto {
            tensor_type: Some(TensorProto {
                dims: vec![-1], // dynamic batch size
                data_type: TensorProtoDataType::Int64 as i32,
                raw_data: vec![],
                name: "".to_string(),
            }),
        };
        let output_type = TypeProto {
            tensor_type: Some(TensorProto {
                dims: vec![-1, dim],
                data_type: TensorProtoDataType::Float as i32,
                raw_data: vec![],
                name: "".to_string(),
            }),
        };

        let graph = GraphProto {
            node: vec![gather_node],
            input: vec![
                ValueInfoProto {
                    name: "input_indices".to_string(),
                    r#type: Some(input_type),
                },
            ],
            output: vec![
                ValueInfoProto {
                    name: "output".to_string(),
                    r#type: Some(output_type),
                },
            ],
            initializer: vec![embedding_tensor],
            name: "embedding_graph".to_string(),
        };

        let opset = OperatorSetIdProto {
            domain: "".to_string(),
            version: 9,
        };

        let model = ModelProto {
            ir_version: 9,
            producer_name: "embedding-trainer".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: "".to_string(),
            graph: Some(graph),
            opset_import: vec![opset],
        };

        let mut buf = Vec::new();
        model.encode(&mut buf).map_err(|e| e.to_string())?;

        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(&buf).map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Saves embeddings in a memory-mappable binary format.
    pub fn save_mmapable_format(&self, path: &str, data: &TrainingData) -> Result<(), String> {
        let words: Vec<String> = data.reverse_vocab.clone();
        let embeddings: Vec<Vec<f32>> = (0..words.len())
            .map(|i| self.embeddings.row(i).to_vec())
            .collect();
        mmap::save_mmapable_format(path, &words, &embeddings)
    }

    /// Loads a memory-mapped embedding file for read-only access.
    pub fn load_mmap(path: &str) -> Result<mmap::MmapEmbeddings, String> {
        mmap::MmapEmbeddings::open(path)
    }
}
