use rand::Rng;
use crate::{EmbeddingModel, TrainingData};

/// Expands a search query with top-k similar words from the embedding space.
pub struct QueryExpander;

impl QueryExpander {
    /// Returns the original query word plus its `top_k` most similar neighbors.
    pub fn expand(
        model: &EmbeddingModel,
        data: &TrainingData,
        query: &str,
        top_k: usize,
    ) -> Vec<String> {
        let mut result = vec![query.to_string()];
        let neighbors = model.semantic_search(query, data, top_k);
        for (word, _) in neighbors {
            result.push(word);
        }
        result
    }
}

/// Agglomerative hierarchical clustering of word embeddings.
pub struct HierarchicalClustering;

impl HierarchicalClustering {
    /// Clusters vocabulary words into `num_clusters` groups using average-linkage
    /// agglomerative clustering on cosine similarity.
    pub fn cluster(
        model: &EmbeddingModel,
        data: &TrainingData,
        num_clusters: usize,
    ) -> Vec<Vec<String>> {
        let n = data.reverse_vocab.len();
        if n == 0 {
            return Vec::new();
        }
        let target_clusters = num_clusters.min(n);

        // Initialize each word as its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        while clusters.len() > target_clusters {
            let mut best_pair = (0, 1);
            let mut best_sim = f32::NEG_INFINITY;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let sim = Self::cluster_similarity(model, &clusters[i], &clusters[j]);
                    if sim > best_sim {
                        best_sim = sim;
                        best_pair = (i, j);
                    }
                }
            }

            let (i, j) = best_pair;
            let mut merged = clusters.remove(j);
            clusters[i].append(&mut merged);
        }

        clusters
            .into_iter()
            .map(|ids| ids.into_iter().map(|id| data.reverse_vocab[id].clone()).collect())
            .collect()
    }

    fn cluster_similarity(model: &EmbeddingModel, a: &[usize], b: &[usize]) -> f32 {
        let mut total_sim = 0.0;
        let mut count = 0usize;
        for &i in a {
            let emb_i = model.embeddings.row(i);
            for &j in b {
                let emb_j = model.embeddings.row(j);
                let dot: f32 = emb_i.iter().zip(emb_j.iter()).map(|(&x, &y)| x * y).sum();
                let norm_i = emb_i.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let norm_j = emb_j.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm_i > 0.0 && norm_j > 0.0 {
                    total_sim += dot / (norm_i * norm_j);
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            total_sim / (count as f32)
        }
    }
}

/// K-means clustering for word embeddings.
pub struct KMeansClustering;

impl KMeansClustering {
    /// Clusters vocabulary words into `k` groups using k-means on embedding vectors.
    ///
    /// Returns a vector of clusters, where each cluster is a list of words.
    pub fn cluster(
        model: &EmbeddingModel,
        data: &TrainingData,
        k: usize,
        max_iterations: usize,
    ) -> Vec<Vec<String>> {
        let n = data.reverse_vocab.len();
        if n == 0 {
            return Vec::new();
        }
        let k = k.min(n);
        let dim = model.config.embedding_dim;

        // Randomly pick k centroids from existing embeddings
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut chosen = std::collections::HashSet::new();
        while centroids.len() < k {
            let idx = rng.gen_range(0..n);
            if chosen.insert(idx) {
                let row = model.embeddings.row(idx);
                centroids.push(row.iter().copied().collect());
            }
        }

        let mut assignments: Vec<usize> = vec![0; n];

        for _ in 0..max_iterations {
            // Assign each point to nearest centroid
            let mut changed = false;
            for (i, assignment) in assignments.iter_mut().enumerate() {
                let emb = model.embeddings.row(i);
                let mut best_dist = f32::INFINITY;
                let mut best_c = 0;
                for (c_idx, centroid) in centroids.iter().enumerate() {
                    let dist: f32 = emb.iter().zip(centroid.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c_idx;
                    }
                }
                if *assignment != best_c {
                    *assignment = best_c;
                    changed = true;
                }
            }
            if !changed {
                break;
            }

            // Recompute centroids
            for (c_idx, centroid) in centroids.iter_mut().enumerate() {
                let mut sum = vec![0.0f32; dim];
                let mut count = 0usize;
                for (i, &assign) in assignments.iter().enumerate() {
                    if assign == c_idx {
                        let emb = model.embeddings.row(i);
                        for j in 0..dim {
                            sum[j] += emb[j];
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    for j in 0..dim {
                        centroid[j] = sum[j] / count as f32;
                    }
                }
            }
        }

        let mut clusters: Vec<Vec<String>> = vec![Vec::new(); k];
        for (i, &assign) in assignments.iter().enumerate() {
            clusters[assign].push(data.reverse_vocab[i].clone());
        }
        clusters.into_iter().filter(|c| !c.is_empty()).collect()
    }
}

/// Locality-Sensitive Hashing index for approximate nearest neighbor search.
pub struct LSHIndex {
    pub hash_tables: Vec<std::collections::HashMap<usize, Vec<usize>>>,
    pub hyperplanes: Vec<Vec<Vec<f32>>>,
    pub num_hashes: usize,
    pub embedding_dim: usize,
}

impl LSHIndex {
    /// Creates a new LSH index with random projection hyperplanes.
    pub fn new(num_hashes: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut hyperplanes = Vec::with_capacity(num_hashes);
        for _ in 0..num_hashes {
            let mut table_planes = Vec::new();
            for _ in 0..32 {
                let plane: Vec<f32> = (0..embedding_dim)
                    .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
                    .collect();
                table_planes.push(plane);
            }
            hyperplanes.push(table_planes);
        }

        let mut hash_tables = Vec::with_capacity(num_hashes);
        for _ in 0..num_hashes {
            hash_tables.push(std::collections::HashMap::new());
        }

        Self {
            hash_tables,
            hyperplanes,
            num_hashes,
            embedding_dim,
        }
    }

    /// Indexes all vocabulary embeddings into the hash tables.
    pub fn build(&mut self, model: &EmbeddingModel, data: &TrainingData) {
        for (word_id, _) in data.reverse_vocab.iter().enumerate() {
            let embedding = model.embeddings.row(word_id);
            for table_id in 0..self.num_hashes {
                let hash = self.compute_hash(&embedding, table_id);
                self.hash_tables[table_id]
                    .entry(hash)
                    .or_default()
                    .push(word_id);
            }
        }
    }

    fn compute_hash(&self, embedding: &ndarray::ArrayView1<f32>, table_id: usize) -> usize {
        let mut hash = 0usize;
        for (bit_idx, plane) in self.hyperplanes[table_id].iter().enumerate() {
            let dot: f32 = embedding.iter().zip(plane.iter()).map(|(&a, &b)| a * b).sum();
            if dot > 0.0 {
                hash |= 1 << bit_idx;
            }
        }
        hash
    }

    /// Approximate top-k nearest neighbors for a query word using LSH.
    pub fn query(&self, query_word: &str, model: &EmbeddingModel, data: &TrainingData, top_k: usize) -> Vec<(String, f32)> {
        let query_emb = match model.get_embedding(query_word, data) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut candidate_set = std::collections::HashSet::new();
        for table_id in 0..self.num_hashes {
            let hash = self.compute_hash(&query_emb.view(), table_id);
            if let Some(bucket) = self.hash_tables[table_id].get(&hash) {
                for &word_id in bucket {
                    candidate_set.insert(word_id);
                }
            }
        }

        let mut results = Vec::new();
        for &word_id in &candidate_set {
            let word = &data.reverse_vocab[word_id];
            if word == query_word {
                continue;
            }
            let candidate = model.embeddings.row(word_id);
            let dot: f32 = query_emb.iter().zip(candidate.iter()).map(|(&a, &b)| a * b).sum();
            let norm_query = query_emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm_candidate = candidate.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_query > 0.0 && norm_candidate > 0.0 {
                results.push((word.clone(), dot / (norm_query * norm_candidate)));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_k).collect()
    }
}
