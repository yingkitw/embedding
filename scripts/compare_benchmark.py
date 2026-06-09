#!/usr/bin/env python3
"""Benchmark comparing embedding-trainer against gensim Word2Vec."""

import subprocess
import tempfile
import time
import json
from pathlib import Path

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


def train_rust(corpus_path: str, dim: int = 100, epochs: int = 5, window: int = 5) -> dict:
    """Train embeddings using the Rust CLI and return timing info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.json"
        emb_path = Path(tmpdir) / "embeddings.txt"

        start = time.perf_counter()
        result = subprocess.run(
            [
                "cargo", "run", "--release", "--",
                "train",
                "--input", corpus_path,
                "--output", str(model_path),
                "--embeddings", str(emb_path),
                "--dim", str(dim),
                "--epochs", str(epochs),
                "--window", str(window),
                "--batch-size", "32",
                "--negative-samples", "5",
                "--model-type", "skipgram",
            ],
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            print("Rust training stderr:", result.stderr)
            raise RuntimeError("Rust training failed")

        return {"time_seconds": elapsed}


def train_gensim(corpus_path: str, dim: int = 100, epochs: int = 5, window: int = 5) -> dict:
    """Train embeddings using gensim Word2Vec and return timing info."""
    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            sentences.append(line.strip().split())

    start = time.perf_counter()
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        sg=1,  # skip-gram
        negative=5,
        epochs=epochs,
        min_count=1,
        workers=1,
    )
    elapsed = time.perf_counter() - start
    return {"time_seconds": elapsed}


def main():
    corpus = Path(__file__).parent.parent / "examples" / "data.txt"
    if not corpus.exists():
        corpus = Path("examples/data.txt")
    if not corpus.exists():
        # Create a synthetic corpus
        corpus = Path("/tmp/benchmark_corpus.txt")
        with open(corpus, "w") as f:
            for _ in range(1000):
                f.write(
                    "the quick brown fox jumps over the lazy dog "
                    "machine learning is fascinating and powerful\n"
                )

    print("=" * 60)
    print("Embedding Benchmark: Rust vs Gensim")
    print("=" * 60)

    rust_result = train_rust(str(corpus))
    print(f"Rust embedding-trainer: {rust_result['time_seconds']:.3f}s")

    if GENSIM_AVAILABLE:
        gensim_result = train_gensim(str(corpus))
        print(f"Gensim Word2Vec:       {gensim_result['time_seconds']:.3f}s")
        ratio = gensim_result["time_seconds"] / rust_result["time_seconds"]
        print(f"Speedup vs gensim:     {ratio:.2f}x")
    else:
        print("(gensim not installed; skipping comparison)")

    print("=" * 60)


if __name__ == "__main__":
    main()
