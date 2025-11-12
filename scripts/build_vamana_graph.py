#!/usr/bin/env python3
"""
Build and visualize a Vamana graph from the Flickr30k multimodal dataset.

This script:
1. Loads image & text pairs from the Flickr30k parquet shards.
2. Uses a CLIP encoder to embed images and captions into a shared vector space.
3. Builds an approximate Vamana graph using inner-product distance.
4. Projects the vectors to 2D (via PCA) and visualizes the graph,
   coloring image nodes red and text nodes blue.

The implementation follows the high-level ideas of the Vamana graph-construction
algorithm (DiskANN) but is simplified for clarity and educational purposes.
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import open_clip
from sklearn.decomposition import PCA


VectorArray = np.ndarray


def inner_product_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance based on negative inner product (lower is closer)."""
    return -float(np.dot(a, b))


def normalize_vectors(vecs: VectorArray) -> VectorArray:
    """L2-normalize vectors to stabilize inner-product comparisons."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def greedy_search(
    data: VectorArray,
    graph: List[Set[int]],
    query_idx: int,
    entry_point: int,
    search_width: int,
) -> List[int]:
    """Greedy graph search used during Vamana construction."""
    if query_idx == entry_point or not graph[entry_point]:
        return [entry_point]

    visited: Set[int] = {entry_point, query_idx}
    candidate_heap: List[Tuple[float, int]] = []
    top_results: List[Tuple[float, int]] = []

    def push_candidate(dist: float, node: int) -> None:
        heapq.heappush(candidate_heap, (dist, node))
        # max-heap via negative distance
        heapq.heappush(top_results, (-dist, node))
        if len(top_results) > search_width:
            heapq.heappop(top_results)

    first_dist = inner_product_distance(data[query_idx], data[entry_point])
    push_candidate(first_dist, entry_point)

    while candidate_heap:
        dist, current = heapq.heappop(candidate_heap)
        worst_top = -top_results[0][0] if top_results else math.inf
        if dist > worst_top:
            break

        for neighbor in graph[current]:
            if neighbor in visited or neighbor == query_idx:
                continue
            visited.add(neighbor)
            neighbor_dist = inner_product_distance(
                data[query_idx], data[neighbor])
            push_candidate(neighbor_dist, neighbor)

    return [node for _, node in sorted([(-d, n) for d, n in top_results])]


def prune_neighbors(
    data: VectorArray,
    node_idx: int,
    candidates: Iterable[int],
    alpha: float,
    max_degree: int,
) -> List[int]:
    """Vamana prune step to retain a diverse set of neighbors."""
    scored_candidates = sorted(
        ((inner_product_distance(data[node_idx], data[cand]), cand)
         for cand in set(candidates) if cand != node_idx),
        key=lambda x: x[0],
    )

    pruned: List[int] = []
    for cand_dist, cand_idx in scored_candidates:
        keep = True
        for existing_idx in pruned:
            dist_existing = inner_product_distance(
                data[cand_idx], data[existing_idx])
            if dist_existing * alpha <= cand_dist:
                keep = False
                break
        if keep:
            pruned.append(cand_idx)
        if len(pruned) >= max_degree:
            break
    return pruned


def build_vamana_graph(
    data: VectorArray,
    entry_point: int,
    max_degree: int = 32,
    search_width: int = 64,
    alpha: float = 1.2,
    seed: int | None = None,
) -> List[Set[int]]:
    """Construct a simplified Vamana graph for the provided dataset."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_points = data.shape[0]
    graph: List[Set[int]] = [set() for _ in range(n_points)]
    insertion_order = list(range(n_points))
    random.shuffle(insertion_order)

    for node_idx in insertion_order:
        if node_idx == entry_point:
            continue

        candidates = greedy_search(
            data=data,
            graph=graph,
            query_idx=node_idx,
            entry_point=entry_point,
            search_width=search_width,
        )
        candidates.extend(graph[node_idx])

        pruned_neighbors = prune_neighbors(
            data=data,
            node_idx=node_idx,
            candidates=candidates,
            alpha=alpha,
            max_degree=max_degree,
        )

        for neighbor in pruned_neighbors:
            graph[node_idx].add(neighbor)
            graph[neighbor].add(node_idx)

            if len(graph[neighbor]) > max_degree:
                updated = prune_neighbors(
                    data=data,
                    node_idx=neighbor,
                    candidates=graph[neighbor],
                    alpha=alpha,
                    max_degree=max_degree,
                )
                graph[neighbor] = set(updated)

        if not graph[node_idx]:
            if node_idx != entry_point:
                graph[node_idx].add(entry_point)
                graph[entry_point].add(node_idx)

    return graph


def project_to_2d(data: VectorArray, seed: int | None = None) -> VectorArray:
    """Project high-dimensional vectors to 2D using PCA."""
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(data)


def visualize_graph(
    graph: Sequence[Set[int]],
    positions_2d: VectorArray,
    labels: Sequence[str],
    max_edges: int = 10000,
    output_path: Path | None = None,
) -> None:
    """Visualize the constructed graph with matplotlib."""
    colors = np.where(np.array(labels) == "image", "red", "blue")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        positions_2d[:, 0],
        positions_2d[:, 1],
        c=colors,
        s=6.0,
        alpha=0.75,
        edgecolors="none",
    )

    edges_plotted = 0
    for src, neighbors in enumerate(graph):
        for dst in neighbors:
            if src >= dst:
                continue
            if edges_plotted >= max_edges:
                break
            xs = [positions_2d[src, 0], positions_2d[dst, 0]]
            ys = [positions_2d[src, 1], positions_2d[dst, 1]]
            ax.plot(xs, ys, color="gray", linewidth=0.3, alpha=0.05)
            edges_plotted += 1

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Vamana Graph (image: red, text: blue)")
    ax.set_aspect("equal", adjustable="datalim")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Graph visualization saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def load_flickr30k_embeddings(
    data_dir: Path,
    count: int,
    device: torch.device,
    caption_index: int = 0,
    batch_size: int = 32,
) -> Tuple[VectorArray, List[str]]:
    """Load Flickr30k samples and return normalized CLIP embeddings with labels."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    image_embeddings: List[np.ndarray] = []
    text_embeddings: List[np.ndarray] = []

    processed = 0
    shard_paths = sorted(data_dir.glob("test-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No parquet shards found in {data_dir}")

    with torch.no_grad():
        for shard_path in shard_paths:
            if processed >= count:
                break

            parquet_file = pq.ParquetFile(shard_path)
            for batch in parquet_file.iter_batches(
                columns=["image", "caption"], batch_size=batch_size
            ):
                batch_images = []
                batch_texts = []
                for image_value, captions in zip(batch["image"], batch["caption"]):
                    if processed >= count:
                        break
                    if image_value is None or captions is None or len(captions) == 0:
                        continue

                    try:
                        image_bytes = image_value["bytes"].as_buffer(
                        ).to_pybytes()
                        image = Image.open(io.BytesIO(
                            image_bytes)).convert("RGB")
                    except Exception:
                        continue

                    caption_idx = min(caption_index, len(captions) - 1)
                    caption = captions[caption_idx].as_py()
                    batch_images.append(preprocess(image))
                    batch_texts.append(caption)
                    processed += 1

                if not batch_images:
                    continue

                image_tensor = torch.stack(batch_images).to(device)
                text_tokens = tokenizer(batch_texts).to(device)

                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tokens)

                image_embeddings.append(
                    normalize_vectors(
                        image_features.cpu().numpy().astype(np.float32))
                )
                text_embeddings.append(
                    normalize_vectors(
                        text_features.cpu().numpy().astype(np.float32))
                )

                if processed >= count:
                    break

    if processed == 0:
        raise RuntimeError("No valid samples loaded from Flickr30k.")

    images = np.vstack(image_embeddings)
    texts = np.vstack(text_embeddings)

    all_vectors = np.vstack([images, texts])
    labels = ["image"] * images.shape[0] + ["text"] * texts.shape[0]
    return all_vectors, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and visualize a Vamana graph for Text2Image data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/vectordbindexing/flickr-30k/data"),
        help="Path to Flickr30k parquet shard directory.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1_000,
        help="Number of image-text pairs to load (total nodes = 2*count).",
    )
    parser.add_argument(
        "--caption-index",
        type=int,
        default=0,
        help="Caption index to use (0-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CLIP embedding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run CLIP encoding on.",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=32,
        help="Maximum degree per node in the Vamana graph.",
    )
    parser.add_argument(
        "--search-width",
        type=int,
        default=64,
        help="Candidate list size during graph construction.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.2,
        help="Pruning diversification parameter.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=10_000,
        help="Maximum number of edges to draw in the visualization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the visualization instead of showing interactively.",
    )

    args = parser.parse_args()

    print(f"Loading {args.count} image-text pairs from {args.data_dir}...")
    device = torch.device(args.device)
    all_vectors, labels = load_flickr30k_embeddings(
        data_dir=args.data_dir,
        count=args.count,
        device=device,
        caption_index=args.caption_index,
        batch_size=args.batch_size,
    )

    entry_point = 0

    print("Building Vamana graph...")
    graph = build_vamana_graph(
        data=all_vectors,
        entry_point=entry_point,
        max_degree=args.max_degree,
        search_width=args.search_width,
        alpha=args.alpha,
        seed=args.seed,
    )

    print("Projecting vectors to 2D for visualization...")
    positions_2d = project_to_2d(all_vectors, seed=args.seed)

    print("Visualizing graph...")
    visualize_graph(
        graph=graph,
        positions_2d=positions_2d,
        labels=labels,
        max_edges=args.max_edges,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
