#!/usr/bin/env python3
"""
Analyze the impact of mixing modalities on Product Quantization (PQ).

Scenario 1:
    - Sample 20k vectors from the base dataset.
    - Train a PQ model, quantize/dequantize those vectors.
    - Compare the distance between two selected base vectors (x_i, x_j)
      before and after quantization.

Scenario 2:
    - Sample 10k base vectors (that include x_i and x_j) and 10k query vectors.
    - Train PQ on the combined 20k vectors, quantize/dequantize them.
    - Compare the distances between x_i and x_j before/after quantization.
    - For a selected query q, compare its distance to x_i before/after quantization.

Distances are computed via inner product to stay consistent with Text2Image usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - safety
    raise SystemExit(
        "faiss is required for this script. Install via `pip install faiss-cpu`."
    ) from exc

from io_utils import read_fbin


def inner_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute inner product distance (larger is closer)."""
    return float(np.dot(a, b))


def load_vectors(path: Path, count: int, start_idx: int = 0) -> np.ndarray:
    """Load `count` vectors from an .fbin file starting at `start_idx`."""
    vectors = read_fbin(str(path), start_idx=start_idx, chunk_size=count)
    return vectors.astype(np.float32, copy=False)


def find_valid_m(dim: int, requested_m: int) -> int:
    """Ensure M divides dimensionality; adjust to closest valid divisor if needed."""
    if dim % requested_m == 0:
        return requested_m

    divisors: List[int] = [d for d in range(1, dim + 1) if dim % d == 0]
    closest = min(divisors, key=lambda d: (abs(d - requested_m), d))
    print(
        f"[Info] Vector dimension {dim} is not divisible by requested M={requested_m}. "
        f"Using closest valid divisor M={closest} instead."
    )
    return closest


def build_pq(
    data: np.ndarray,
    m_subvectors: int,
    nbits: int,
    training_iterations: int,
) -> faiss.ProductQuantizer:
    """Train a ProductQuantizer and return the trained instance."""
    d = data.shape[1]
    pq = faiss.ProductQuantizer(d, m_subvectors, nbits)

    # Configure extra training iterations if supported (newer Faiss)
    if hasattr(pq, "train_type") and hasattr(faiss, "Train_type"):
        pq.train_type = faiss.Train_type.Clustering
    if hasattr(pq, "set_verbosity"):
        pq.set_verbosity(1)

    # Clamp training iterations via the clustering parameters
    if hasattr(pq, "cp"):
        pq.cp.niter = training_iterations

    cvar = getattr(faiss, "cvar", None)
    if cvar is not None and hasattr(cvar, "pq_train_type"):
        cvar.pq_train_type = 0  # Clustering-based training
    pq.train(data)
    return pq


def quantize_and_reconstruct(
    pq: faiss.ProductQuantizer,
    data: np.ndarray,
) -> np.ndarray:
    """Encode data with PQ and decode back to reconstructed vectors."""
    codes = pq.compute_codes(data)
    reconstructed = pq.decode(codes)
    return reconstructed


def scenario_one(
    base_vectors: np.ndarray,
    m_subvectors: int,
    nbits: int,
    training_iterations: int,
    index_i: int,
    index_j: int,
) -> Tuple[float, float]:
    """Run Scenario 1 and return original vs reconstructed distances."""
    pq = build_pq(base_vectors, m_subvectors, nbits, training_iterations)
    reconstructed = quantize_and_reconstruct(pq, base_vectors)

    dist_original = inner_product(base_vectors[index_i], base_vectors[index_j])
    dist_reconstructed = inner_product(
        reconstructed[index_i], reconstructed[index_j])
    return dist_original, dist_reconstructed


def scenario_two(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    m_subvectors: int,
    nbits: int,
    training_iterations: int,
    index_i: int,
    index_j: int,
    query_index: int,
) -> Tuple[float, float, float, float]:
    """Run Scenario 2, returning distances for x_i/x_j and q/x_i."""
    combined = np.vstack([base_vectors, query_vectors]
                         ).astype(np.float32, copy=False)
    pq = build_pq(combined, m_subvectors, nbits, training_iterations)
    reconstructed = quantize_and_reconstruct(pq, combined)

    dist_orig_xixj = inner_product(
        base_vectors[index_i], base_vectors[index_j])
    dist_recon_xixj = inner_product(
        reconstructed[index_i], reconstructed[index_j]
    )

    query_offset = base_vectors.shape[0]
    orig_query_vector = query_vectors[query_index]
    recon_query_vector = reconstructed[query_offset + query_index]

    dist_orig_q_xi = inner_product(orig_query_vector, base_vectors[index_i])
    dist_recon_q_xi = inner_product(
        recon_query_vector,
        reconstructed[index_i],
    )
    return dist_orig_xixj, dist_recon_xixj, dist_orig_q_xi, dist_recon_q_xi


def validate_indices(
    base_count: int,
    query_count: int,
    index_i: int,
    index_j: int,
    query_index: int,
) -> None:
    """Ensure provided indices fall within available data."""
    if not (0 <= index_i < base_count and 0 <= index_j < base_count):
        raise ValueError(
            "x_i and x_j indices must lie within the selected base vectors.")
    if not 0 <= query_index < query_count:
        raise ValueError(
            "Query index must lie within the selected query vectors.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PQ quantization effects under unimodal vs multimodal data."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("/workspace/vectordbindexing/Text2Image/base.10M.fbin"),
        help="Path to base dataset .fbin file.",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        default=Path(
            "/workspace/vectordbindexing/Text2Image/query.train.10M.fbin"),
        help="Path to query dataset .fbin file.",
    )
    parser.add_argument(
        "--scenario1-count",
        type=int,
        default=200000,
        help="Number of base vectors to load for Scenario 1.",
    )
    parser.add_argument(
        "--scenario2-base-count",
        type=int,
        default=100000,
        help="Number of base vectors to load for Scenario 2.",
    )
    parser.add_argument(
        "--scenario2-query-count",
        type=int,
        default=100000,
        help="Number of query vectors to load for Scenario 2.",
    )
    parser.add_argument(
        "--m-subvectors",
        type=int,
        default=16,
        help="Number of PQ subvectors (M).",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=8,
        help="Bits per subvector code.",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=100,
        help="Number of k-means iterations used during PQ training.",
    )
    parser.add_argument(
        "--index-i",
        type=int,
        default=0,
        help="Index of x_i in the selected base vectors.",
    )
    parser.add_argument(
        "--index-j",
        type=int,
        default=1,
        help="Index of x_j in the selected base vectors.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Index of q in the selected query vectors (Scenario 2).",
    )
    parser.add_argument(
        "--base-start",
        type=int,
        default=0,
        help="Starting offset when sampling base vectors.",
    )
    parser.add_argument(
        "--query-start",
        type=int,
        default=0,
        help="Starting offset when sampling query vectors.",
    )
    args = parser.parse_args()

    if args.index_i == args.index_j:
        raise SystemExit(
            "index-i and index-j must refer to different vectors.")

    # Scenario 1 relies on the first `scenario1_count` vectors.
    print(
        f"[Scenario 1] Loading {args.scenario1_count} base vectors from {args.base_path}...")
    scenario1_base = load_vectors(
        args.base_path, args.scenario1_count, args.base_start)
    vector_dim = scenario1_base.shape[1]
    m_subvectors = find_valid_m(vector_dim, args.m_subvectors)

    validate_indices(
        base_count=args.scenario2_base_count,
        query_count=args.scenario2_query_count,
        index_i=args.index_i,
        index_j=args.index_j,
        query_index=args.query_index,
    )

    print("[Scenario 1] Training PQ and computing distances...")
    s1_orig, s1_recon = scenario_one(
        base_vectors=scenario1_base,
        m_subvectors=m_subvectors,
        nbits=args.nbits,
        training_iterations=args.training_iterations,
        index_i=args.index_i,
        index_j=args.index_j,
    )

    print(f"[Scenario 1] Inner product(x_i, x_j) original    : {s1_orig:.6f}")
    print(
        f"[Scenario 1] Inner product(x_i, x_j) reconstructed: {s1_recon:.6f}")

    # Scenario 2 uses subset of base vectors that must include x_i and x_j
    print(
        f"\n[Scenario 2] Loading {args.scenario2_base_count} base vectors...")
    scenario2_base = load_vectors(
        args.base_path,
        args.scenario2_base_count,
        args.base_start,
    )

    print(
        f"[Scenario 2] Loading {args.scenario2_query_count} query vectors from {args.query_path}...")
    scenario2_query = load_vectors(
        args.query_path,
        args.scenario2_query_count,
        args.query_start,
    )

    if scenario2_query.shape[1] != vector_dim:
        raise SystemExit(
            f"Dimension mismatch: base vectors dim={vector_dim}, "
            f"query vectors dim={scenario2_query.shape[1]}."
        )

    print("[Scenario 2] Training PQ on combined base + query vectors...")
    (
        s2_orig_xixj,
        s2_recon_xixj,
        s2_orig_q_xi,
        s2_recon_q_xi,
    ) = scenario_two(
        base_vectors=scenario2_base,
        query_vectors=scenario2_query,
        m_subvectors=m_subvectors,
        nbits=args.nbits,
        training_iterations=args.training_iterations,
        index_i=args.index_i,
        index_j=args.index_j,
        query_index=args.query_index,
    )

    print(
        f"[Scenario 2] Inner product(x_i, x_j) original    : {s2_orig_xixj:.6f}")
    print(
        f"[Scenario 2] Inner product(x_i, x_j) reconstructed: {s2_recon_xixj:.6f}")
    print(
        f"[Scenario 2] Inner product(q, x_i) original       : {s2_orig_q_xi:.6f}")
    print(
        f"[Scenario 2] Inner product(q, x_i) reconstructed  : {s2_recon_q_xi:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - convenience
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
