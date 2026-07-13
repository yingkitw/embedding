use ndarray::Array2;
use serde::{Deserialize, Serialize};
use half::f16;
use prost::Message;

use crate::config::TrainingData;
use crate::model::EmbeddingModel;
use crate::onnx::*;

/// Quantization precision for embedding compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// 8-bit signed integer with per-row symmetric scaling (~4x smaller).
    Int8,
    /// IEEE 754 half precision (~2x smaller).
    Fp16,
}

/// Post-training quantized embedding storage with dequantization support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedEmbeddings {
    pub mode: QuantizationMode,
    pub vocab_size: usize,
    pub dim: usize,
    int8_data: Vec<i8>,
    int8_scales: Vec<f32>,
    fp16_data: Vec<u16>,
}

impl QuantizedEmbeddings {
    /// Quantizes an embedding matrix using the chosen precision.
    pub fn from_array(embeddings: &Array2<f32>, mode: QuantizationMode) -> Self {
        let (vocab_size, dim) = embeddings.dim();
        match mode {
            QuantizationMode::Int8 => {
                let (int8_data, int8_scales) = quantize_int8_per_row(embeddings);
                Self {
                    mode,
                    vocab_size,
                    dim,
                    int8_data,
                    int8_scales,
                    fp16_data: Vec::new(),
                }
            }
            QuantizationMode::Fp16 => {
                let fp16_data = quantize_fp16(embeddings);
                Self {
                    mode,
                    vocab_size,
                    dim,
                    int8_data: Vec::new(),
                    int8_scales: Vec::new(),
                    fp16_data,
                }
            }
        }
    }

    /// Quantizes a trained model's embedding matrix.
    pub fn from_model(model: &EmbeddingModel, mode: QuantizationMode) -> Self {
        Self::from_array(&model.embeddings, mode)
    }

    /// Returns the dequantized embedding vector for a vocabulary row.
    pub fn dequantize_row(&self, row: usize) -> Option<Vec<f32>> {
        if row >= self.vocab_size {
            return None;
        }
        Some(match self.mode {
            QuantizationMode::Int8 => dequantize_int8_row(
                &self.int8_data,
                self.int8_scales[row],
                row,
                self.dim,
            ),
            QuantizationMode::Fp16 => dequantize_fp16_row(&self.fp16_data, row, self.dim),
        })
    }

    /// Looks up a word embedding by dequantizing its stored row.
    pub fn get(&self, word: &str, data: &TrainingData) -> Option<Vec<f32>> {
        let id = data.vocab.get(word)?;
        self.dequantize_row(*id)
    }

    /// Cosine similarity between two words using dequantized vectors.
    pub fn similarity(&self, word1: &str, word2: &str, data: &TrainingData) -> Option<f32> {
        let v1 = self.get(word1, data)?;
        let v2 = self.get(word2, data)?;
        Some(cosine_similarity(&v1, &v2))
    }

    /// Approximate size reduction factor relative to f32 storage.
    pub fn compression_ratio(&self) -> f32 {
        match self.mode {
            QuantizationMode::Int8 => 4.0,
            QuantizationMode::Fp16 => 2.0,
        }
    }

    /// Mean absolute error between dequantized and original embeddings.
    pub fn reconstruction_error(&self, original: &Array2<f32>) -> f32 {
        let (rows, cols) = original.dim();
        assert_eq!(rows, self.vocab_size);
        assert_eq!(cols, self.dim);

        let mut total = 0.0f32;
        let mut count = 0usize;
        for row in 0..rows {
            let orig = original.row(row);
            let recon = self.dequantize_row(row).unwrap();
            for (a, b) in orig.iter().zip(recon.iter()) {
                total += (a - b).abs();
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    /// Saves a quantized ONNX model with a Gather lookup node.
    pub fn save_onnx(&self, path: &str, data: &TrainingData) -> Result<(), String> {
        let vocab_size = data.reverse_vocab.len() as i64;
        let dim = self.dim as i64;

        let (data_type, raw_data, output_type) = match self.mode {
            QuantizationMode::Int8 => {
                let raw: Vec<u8> = self.int8_data.iter().map(|&b| b as u8).collect();
                (
                    TensorProtoDataType::Int8 as i32,
                    raw,
                    TensorProtoDataType::Int8 as i32,
                )
            }
            QuantizationMode::Fp16 => {
                let mut raw = Vec::with_capacity(self.fp16_data.len() * 2);
                for &bits in &self.fp16_data {
                    raw.extend_from_slice(&bits.to_le_bytes());
                }
                (
                    TensorProtoDataType::Float16 as i32,
                    raw,
                    TensorProtoDataType::Float16 as i32,
                )
            }
        };

        let embedding_tensor = TensorProto {
            dims: vec![vocab_size, dim],
            data_type,
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
                dims: vec![-1],
                data_type: TensorProtoDataType::Int64 as i32,
                raw_data: vec![],
                name: "".to_string(),
            }),
        };
        let output_type_proto = TypeProto {
            tensor_type: Some(TensorProto {
                dims: vec![-1, dim],
                data_type: output_type,
                raw_data: vec![],
                name: "".to_string(),
            }),
        };

        let graph = GraphProto {
            node: vec![gather_node],
            input: vec![ValueInfoProto {
                name: "input_indices".to_string(),
                r#type: Some(input_type),
            }],
            output: vec![ValueInfoProto {
                name: "output".to_string(),
                r#type: Some(output_type_proto),
            }],
            initializer: vec![embedding_tensor],
            name: "quantized_embedding_graph".to_string(),
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
        std::fs::write(path, &buf).map_err(|e| e.to_string())?;
        Ok(())
    }
}

impl EmbeddingModel {
    /// Post-training quantization of the embedding matrix.
    pub fn quantize(&self, mode: QuantizationMode) -> QuantizedEmbeddings {
        QuantizedEmbeddings::from_model(self, mode)
    }

    /// Saves a quantized ONNX model (INT8 or FP16 weights).
    pub fn save_quantized_onnx_format(
        &self,
        path: &str,
        data: &TrainingData,
        mode: QuantizationMode,
    ) -> Result<(), String> {
        self.quantize(mode).save_onnx(path, data)
    }
}

fn quantize_int8_per_row(embeddings: &Array2<f32>) -> (Vec<i8>, Vec<f32>) {
    let (rows, cols) = embeddings.dim();
    let mut data = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows);

    for row in embeddings.rows() {
        let max_abs = row.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.push(scale);
        for &x in row.iter() {
            let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
            data.push(q);
        }
    }
    (data, scales)
}

fn dequantize_int8_row(data: &[i8], scale: f32, row: usize, dim: usize) -> Vec<f32> {
    let start = row * dim;
    data[start..start + dim]
        .iter()
        .map(|&q| q as f32 * scale)
        .collect()
}

fn quantize_fp16(embeddings: &Array2<f32>) -> Vec<u16> {
    embeddings
        .iter()
        .map(|&x| f16::from_f32(x).to_bits())
        .collect()
}

fn dequantize_fp16_row(data: &[u16], row: usize, dim: usize) -> Vec<f32> {
    let start = row * dim;
    data[start..start + dim]
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelType, TrainingConfig, TrainingData};
    use prost::Message;

    fn make_trained_model() -> (EmbeddingModel, TrainingData) {
        let data = TrainingData::from_text(
            "the cat sat on the mat. the dog sat on the log. the cat chased the dog.",
        );
        let config = TrainingConfig::new(ModelType::SkipGram)
            .with_dim(8)
            .with_epochs(2);
        let mut model = EmbeddingModel::new(config, data.vocab.len());
        model.train(&data).unwrap();
        (model, data)
    }

    #[test]
    fn test_int8_quantization_roundtrip() {
        let (model, data) = make_trained_model();
        let q = model.quantize(QuantizationMode::Int8);

        assert_eq!(q.vocab_size, model.vocab_size);
        assert_eq!(q.dim, model.config.embedding_dim);
        assert_eq!(q.compression_ratio(), 4.0);

        let cat_id = data.vocab["cat"];
        let orig: Vec<f32> = model.embeddings.row(cat_id).to_vec();
        let recon = q.dequantize_row(cat_id).unwrap();

        for (a, b) in orig.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 0.1, "INT8 error too large: {} vs {}", a, b);
        }

        let err = q.reconstruction_error(&model.embeddings);
        assert!(err < 0.05, "Mean reconstruction error: {}", err);
    }

    #[test]
    fn test_fp16_quantization_roundtrip() {
        let (model, data) = make_trained_model();
        let q = model.quantize(QuantizationMode::Fp16);

        assert_eq!(q.compression_ratio(), 2.0);

        let cat_id = data.vocab["cat"];
        let orig: Vec<f32> = model.embeddings.row(cat_id).to_vec();
        let recon = q.dequantize_row(cat_id).unwrap();

        for (a, b) in orig.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 1e-3, "FP16 error too large: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_quantized_similarity() {
        let (model, data) = make_trained_model();
        let q = model.quantize(QuantizationMode::Int8);

        let orig = model.similarity("cat", "dog", &data);
        let quant = q.similarity("cat", "dog", &data);
        assert!(orig.is_some() && quant.is_some());
        assert!((orig.unwrap() - quant.unwrap()).abs() < 0.15);
    }

    #[test]
    fn test_save_quantized_onnx_int8() {
        let (model, data) = make_trained_model();
        let path = std::env::temp_dir().join("test_quant_int8.onnx");
        let path_str = path.to_str().unwrap();

        model
            .save_quantized_onnx_format(path_str, &data, QuantizationMode::Int8)
            .unwrap();

        let bytes = std::fs::read(path_str).unwrap();
        let decoded = ModelProto::decode(&bytes[..]).unwrap();
        let graph = decoded.graph.unwrap();
        assert_eq!(graph.initializer[0].data_type, TensorProtoDataType::Int8 as i32);

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_save_quantized_onnx_fp16() {
        let (model, data) = make_trained_model();
        let path = std::env::temp_dir().join("test_quant_fp16.onnx");
        let path_str = path.to_str().unwrap();

        model
            .save_quantized_onnx_format(path_str, &data, QuantizationMode::Fp16)
            .unwrap();

        let bytes = std::fs::read(path_str).unwrap();
        let decoded = ModelProto::decode(&bytes[..]).unwrap();
        let graph = decoded.graph.unwrap();
        assert_eq!(
            graph.initializer[0].data_type,
            TensorProtoDataType::Float16 as i32
        );

        std::fs::remove_file(path_str).ok();
    }
}
