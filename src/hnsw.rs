use rand::Rng;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use crate::{EmbeddingModel, TrainingData};

/// Hierarchical Navigable Small World index for approximate nearest neighbor search.
///
/// HNSW typically achieves higher recall than LSH at comparable latency and scales
/// to much larger vocabularies. Use [`HNSWIndex::new`] with default parameters for
/// most workloads.
pub struct HNSWIndex {
    nodes: Vec<HNSWNode>,
    entry_point: Option<usize>,
    max_layer: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f64,
}

struct HNSWNode {
    word_id: usize,
    neighbors: Vec<Vec<usize>>,
}

/// Total ordering wrapper for f32 distances in priority queues.
#[derive(Copy, Clone, PartialEq)]
struct Dist(f32);

impl Eq for Dist {}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
    }
}

impl HNSWIndex {
    /// Creates an HNSW index with the given graph parameters.
    ///
    /// - `m`: max bidirectional links per node (layers > 0)
    /// - `ef_construction`: candidate list size during index build
    /// - `ef_search`: candidate list size during query
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let m = m.max(2);
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m_max0: m * 2,
            ef_construction: ef_construction.max(m),
            ef_search: ef_search.max(m),
            ml: 1.0 / (m as f64).ln(),
        }
    }

    /// Creates an index with sensible defaults (`m=16`, `ef_construction=100`, `ef_search=50`).
    pub fn with_defaults() -> Self {
        Self::new(16, 100, 50)
    }

    /// Inserts all vocabulary embeddings into the graph.
    pub fn build(&mut self, model: &EmbeddingModel, data: &TrainingData) {
        for word_id in 0..data.reverse_vocab.len() {
            self.insert(model, word_id);
        }
    }

    /// Approximate top-k nearest neighbors for a query word.
    pub fn query(
        &self,
        query_word: &str,
        model: &EmbeddingModel,
        data: &TrainingData,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let query_emb = match model.get_embedding(query_word, data) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let query_id = data.vocab.get(query_word).copied();

        let neighbors = self.search(query_emb.as_slice().unwrap(), model, top_k + 1, self.ef_search);
        neighbors
            .into_iter()
            .filter(|(id, _)| {
                Some(*id) != query_id && data.reverse_vocab[*id] != query_word
            })
            .take(top_k)
            .map(|(id, sim)| (data.reverse_vocab[id].clone(), sim))
            .collect()
    }

    /// Returns the number of indexed nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the index contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn insert(&mut self, model: &EmbeddingModel, word_id: usize) {
        let mut rng = rand::thread_rng();
        let level = self.random_level(&mut rng);
        let node_idx = self.nodes.len();

        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }

        self.nodes.push(HNSWNode {
            word_id,
            neighbors,
        });

        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_layer = level;
            return;
        }

        let query = model.embeddings.row(word_id).to_vec();
        let mut current_ep = self.entry_point.unwrap();

        for lc in (level + 1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(&query, model, current_ep, lc);
        }

        let start_layer = level.min(self.max_layer);
        for lc in (0..=start_layer).rev() {
            let candidates = self.search_layer(&query, model, current_ep, self.ef_construction, lc);
            let selected = self.select_neighbors(&query, model, &candidates, self.layer_m(lc), node_idx);

            for &neighbor_idx in &selected {
                self.connect(node_idx, neighbor_idx, lc, model);
            }

            if !selected.is_empty() {
                current_ep = selected[0];
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(node_idx);
        }
    }

    fn search(
        &self,
        query: &[f32],
        model: &EmbeddingModel,
        top_k: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let mut current = self.entry_point.unwrap();
        for lc in (1..=self.max_layer).rev() {
            current = self.search_layer_single(query, model, current, lc);
        }

        let candidates = self.search_layer(query, model, current, ef.max(top_k), 0);
        candidates
            .into_iter()
            .take(top_k)
            .map(|(idx, dist)| (self.nodes[idx].word_id, 1.0 - dist))
            .collect()
    }

    fn search_layer_single(
        &self,
        query: &[f32],
        model: &EmbeddingModel,
        entry: usize,
        layer: usize,
    ) -> usize {
        let mut best = entry;
        let mut best_dist = self.distance(query, model, entry);

        loop {
            let mut improved = false;
            for &neighbor in &self.nodes[best].neighbors[layer] {
                let dist = self.distance(query, model, neighbor);
                if dist < best_dist {
                    best_dist = dist;
                    best = neighbor;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    fn search_layer(
        &self,
        query: &[f32],
        model: &EmbeddingModel,
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        visited.insert(entry);

        let entry_dist = self.distance(query, model, entry);
        let mut candidates = BinaryHeap::new();
        candidates.push((Reverse(Dist(entry_dist)), entry));

        let mut results = BinaryHeap::new();
        results.push((Dist(entry_dist), entry));

        while let Some((Reverse(Dist(c_dist)), current)) = candidates.pop() {
            let worst_result = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);
            if c_dist > worst_result && results.len() >= ef {
                break;
            }

            for &neighbor in &self.nodes[current].neighbors[layer] {
                if !visited.insert(neighbor) {
                    continue;
                }
                let dist = self.distance(query, model, neighbor);

                if results.len() < ef || dist < results.peek().unwrap().0.0 {
                    candidates.push((Reverse(Dist(dist)), neighbor));
                    results.push((Dist(dist), neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out
    }

    fn select_neighbors(
        &self,
        _query: &[f32],
        _model: &EmbeddingModel,
        candidates: &[(usize, f32)],
        m: usize,
        exclude: usize,
    ) -> Vec<usize> {
        let mut sorted: Vec<(usize, f32)> = candidates
            .iter()
            .copied()
            .filter(|(id, _)| *id != exclude)
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.into_iter().take(m).map(|(id, _)| id).collect()
    }

    fn connect(&mut self, a: usize, b: usize, layer: usize, model: &EmbeddingModel) {
        if !self.nodes[a].neighbors[layer].contains(&b) {
            self.nodes[a].neighbors[layer].push(b);
        }
        if !self.nodes[b].neighbors[layer].contains(&a) {
            self.nodes[b].neighbors[layer].push(a);
        }
        self.shrink(a, layer, model);
        self.shrink(b, layer, model);
    }

    fn shrink(&mut self, node_idx: usize, layer: usize, model: &EmbeddingModel) {
        let max_conn = self.layer_m(layer);
        if self.nodes[node_idx].neighbors[layer].len() <= max_conn {
            return;
        }

        let emb = model.embeddings.row(self.nodes[node_idx].word_id);
        let mut neighbors: Vec<(usize, f32)> = self.nodes[node_idx].neighbors[layer]
            .iter()
            .map(|&n| {
                let n_emb = model.embeddings.row(self.nodes[n].word_id);
                (n, cosine_distance_views(emb.view(), n_emb.view()))
            })
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        self.nodes[node_idx].neighbors[layer] =
            neighbors.into_iter().take(max_conn).map(|(n, _)| n).collect();
    }

    fn distance(&self, query: &[f32], model: &EmbeddingModel, node_idx: usize) -> f32 {
        let word_id = self.nodes[node_idx].word_id;
        cosine_distance_query(query, model.embeddings.row(word_id))
    }

    fn layer_m(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max0
        } else {
            self.m
        }
    }

    fn random_level(&self, rng: &mut impl Rng) -> usize {
        let mut level = 0usize;
        while rng.r#gen::<f64>() < self.ml && level < 16 {
            level += 1;
        }
        level
    }
}

fn cosine_distance_views(a: ndarray::ArrayView1<f32>, b: ndarray::ArrayView1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

fn cosine_distance_query(query: &[f32], b: ndarray::ArrayView1<f32>) -> f32 {
    let dot: f32 = query.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelType, TrainingConfig};

    fn make_trained_model() -> (EmbeddingModel, TrainingData) {
        let data = TrainingData::from_text(
            "the cat sat on the mat. the dog sat on the log. the cat chased the dog. \
             fox jumps over lazy dog. machine learning uses embeddings.",
        );
        let config = TrainingConfig::new(ModelType::SkipGram)
            .with_dim(16)
            .with_epochs(5);
        let mut model = EmbeddingModel::new(config, data.vocab.len());
        model.train(&data).unwrap();
        (model, data)
    }

    #[test]
    fn test_hnsw_build_and_query() {
        let (model, data) = make_trained_model();
        let mut index = HNSWIndex::with_defaults();
        index.build(&model, &data);

        assert_eq!(index.len(), data.vocab.len());
        let results = index.query("cat", &model, &data, 5);
        assert!(!results.is_empty());
        for (word, sim) in &results {
            assert_ne!(word, "cat");
            assert!(*sim >= -1.0 && *sim <= 1.0);
        }
    }

    #[test]
    fn test_hnsw_recall_vs_exact_search() {
        let (model, data) = make_trained_model();
        let mut index = HNSWIndex::with_defaults();
        index.build(&model, &data);

        for query in ["cat", "dog", "fox"] {
            let exact = model.semantic_search(query, &data, 5);
            let approx = index.query(query, &model, &data, 5);
            assert!(!exact.is_empty(), "exact search empty for {}", query);
            assert!(!approx.is_empty(), "HNSW search empty for {}", query);

            let exact_words: HashSet<_> = exact.iter().map(|(w, _)| w.as_str()).collect();
            let overlap = approx
                .iter()
                .filter(|(w, _)| exact_words.contains(w.as_str()))
                .count();
            assert!(
                overlap >= 1,
                "HNSW should recall at least one exact top-5 neighbor for '{}'",
                query
            );
        }
    }

    #[test]
    fn test_hnsw_unknown_query() {
        let (model, data) = make_trained_model();
        let mut index = HNSWIndex::with_defaults();
        index.build(&model, &data);
        assert!(index.query("nonexistent", &model, &data, 5).is_empty());
    }
}
