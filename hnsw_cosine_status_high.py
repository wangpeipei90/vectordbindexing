# from __future__ import annotations

"""
HNSW Index with High Layer Edge Addition

This file is a modified version of hnsw_cosine_status.py with the following key changes:
1. build_cross_distribution_edges() now only supports layers >= 1 (not layer 0)
2. augment_from_query_topk() defaults to layer 1 instead of layer 0
3. All edge addition logic has been modified to avoid adding edges on layer 0
4. This ensures that new edges are concentrated on higher layers, not the base layer

The original file concentrated new edges on layer 0, but this version distributes
them across higher layers to avoid conflicts with the base layer structure.
"""

import math
import random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

from simple_sim_hash import SimpleSimHash


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算L2距离（欧氏距离）"""
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def l2_distance_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    批量计算L2距离
    a: shape (d,) - 单个向量
    b: shape (n, d) - 多个向量
    返回: shape (n,) - 距离数组
    """
    # 广播计算差值
    diff = b - a.reshape(1, -1)  # shape (n, d)
    # 计算L2距离
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    return distances


@dataclass
class Node:
    """Represents a single data item in the HNSW index.

    Attributes
    ----------
    vector : int
        The binary vector encoded as an integer.
    level : int
        The maximum layer to which this node belongs (layer 0 is the base).
    """

    vector: np.ndarray
    level: int


class HNSWIndex:
    """Hierarchical Navigable Small World index using L2 distance.

    Parameters
    ----------
    M : int, optional
        Max number of neighbours per node in each layer (default 8).
    ef_construction : int, optional
        Candidate list size during construction (default 200).
    ef_search : int, optional
        Candidate list size during search (default 50).
    random_seed : Optional[int], optional
        Random seed for reproducibility.
        distance_fn : callable, optional
        Distance function; default is l2_distance.
    """

    def __init__(
        self,
        M: int = 8,
        ef_construction: int = 200,
        ef_search: int = 50,
        random_seed: Optional[int] = None,
        distance_fn=l2_distance,
    ) -> None:
        self.M = M
        self.ef_construction = max(ef_construction, M)
        self.ef_search = max(ef_search, M)
        self.distance = distance_fn

        # map node id -> Node
        self.items: Dict[int, Node] = {}
        # adjacency lists per layer: layer -> id -> list of neighbour ids
        self.neighbours: Dict[int, Dict[int, List[int]]
                              ] = defaultdict(lambda: defaultdict(list))
        # 新增：边的标记信息，用于跟踪 cross distribution 边
        self.edge_flags: Dict[int, Dict[int, Dict[int, str]]
                              ] = defaultdict(lambda: defaultdict(dict))
        # 新增：统计信息
        self.cross_distribution_stats = {
            "total_cross_edges": 0,
            "deleted_cross_edges": 0,
            "cross_edges_by_query": defaultdict(int)
        }
        self.max_level: int = -1
        self.entry_point: Optional[int] = None
        self._id_counter = 0
        if random_seed is not None:
            random.seed(random_seed)

    def _assign_level(self) -> int:
        """Draw a random level for a new node (exponential distribution)."""
        if self.M <= 1:
            return 0
        lam = 1.0 / math.log(self.M)
        r = random.random()
        level = int(-math.log(r) * lam)
        return level

    def select_candidate_ids(self, vec: np.ndarray, limit: int = 10000,
                             lsh: Optional[SimpleSimHash] = None,
                             upper_layer_budget: int = 5000) -> List[int]:
        cand = []

        # A) 上层引导：从最高层做有界扩展，逐层向下累积
        visited = set()
        frontier = [self.entry_point] if self.entry_point is not None else []
        for L in range(self.max_level, -1, -1):
            next_frontier = []
            steps = 0
            while frontier and steps < upper_layer_budget // max(1, (self.max_level+1)):
                nid = frontier.pop()
                if nid in visited:
                    continue
                visited.add(nid)
                cand.append(nid)
                # 邻居推进
                for nb in self.neighbours[L][nid]:
                    if nb not in visited:
                        next_frontier.append(nb)
                steps += 1
            frontier = next_frontier
            if len(cand) >= limit:
                break

        # B) LSH 补充（可选）
        if lsh is not None and len(cand) < limit:
            lsh_ids = lsh.get_near(vec, radius=1)
            # 用真距离重排，补齐到 limit
            if lsh_ids:
                # 去重
                pool = list(set(lsh_ids).difference(cand))
                if pool:
                    item_pool = [self.items[idx].vector for idx in pool]
                    d = l2_distance_batch(vec, np.array(item_pool))
                    take = min(limit - len(cand), len(pool))
                    idx = np.argpartition(d, take-1)[:take]
                    cand.extend([pool[i] for i in idx.tolist()])

        # 截断到 limit
        if len(cand) > limit:
            cand = cand[:limit]
        return cand

    # ==== 1) 只在白名单里搜索（greedy / single-layer） ====
    def _search_layer_greedy_allowed(self, vec: np.ndarray, entry_id: int, layer: int,
                                     allowed: Optional[set] = None) -> int:
        current = entry_id
        best_dist = self.distance(vec, self.items[current].vector)
        improved = True
        while improved:
            improved = False
            nbs = self.neighbours[layer][current]
            if allowed is not None:
                nbs = [nb for nb in nbs if nb in allowed]
            if not nbs:
                break
            item_nbs = [self.items[idx].vector for idx in nbs]
            dists = l2_distance_batch(vec, np.array(item_nbs))
            j = int(np.argmin(dists))
            if dists[j] < best_dist:
                best_dist = float(dists[j])
                current = nbs[j]
                improved = True
        return current

    def _search_layer_allowed(self, vec: np.ndarray, entry_id: int, layer: int, ef: int,
                              allowed: Optional[set] = None) -> List[int]:
        import heapq
        import math
        N = self._id_counter
        visited = np.zeros(N, dtype=bool)
        candidates: List[Tuple[float, int]] = []
        result: List[Tuple[float, int]] = []

        dist_entry = self.distance(vec, self.items[entry_id].vector)
        heapq.heappush(candidates, (dist_entry, entry_id))
        heapq.heappush(result, (-dist_entry, entry_id))
        visited[entry_id] = True

        while candidates:
            dist_curr, curr = heapq.heappop(candidates)
            worst_dist = -result[0][0] if result else math.inf
            if dist_curr > worst_dist:
                break

            nbs = self.neighbours[layer][curr]
            if allowed is not None:
                nbs = [nb for nb in nbs if not visited[nb] and nb in allowed]
            else:
                nbs = [nb for nb in nbs if not visited[nb]]
            if not nbs:
                continue

            item_nbs = [self.items[idx].vector for idx in nbs]
            dvec = l2_distance_batch(vec, np.array(item_nbs))
            if len(dvec) > ef:
                idx = np.argpartition(dvec, ef-1)[:ef]
                nbs = [nbs[i] for i in idx.tolist()]
                dvec = dvec[idx]

            for nb, d in zip(nbs, dvec):
                visited[nb] = True
                d = float(d)
                if len(result) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(result, (-d, nb))
                    if len(result) > ef:
                        heapq.heappop(result)
                        worst_dist = -result[0][0]
        return [nid for (_, nid) in result]

    def add_item_fast10k(self, vector: np.ndarray, id: Optional[int] = None,
                         lsh: Optional[SimpleSimHash] = None, limit: int = 10000) -> int:
        # L2距离不需要归一化，直接使用原始向量
        vec = np.asarray(vector, dtype=np.float32)
        if id is None:
            id = self._id_counter
            self._id_counter += 1
        else:
            # 当传入显式ID时，确保_id_counter足够大
            self._id_counter = max(self._id_counter, id + 1)
        level = self._assign_level()
        self.items[id] = Node(vector=vec, level=level)

        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            if lsh is not None:
                lsh.add(id, vec)
            return id

        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

        # 生成候选池（最多1万）
        allowed_ids = set(self.select_candidate_ids(vec, limit=limit, lsh=lsh))

        # 自顶向下：greedy 限于 allowed
        current_node = self.entry_point
        for l in range(self.max_level, level, -1):
            current_node = self._search_layer_greedy_allowed(
                vec, current_node, l, allowed_ids)

        # 底层及以下各层：单层 best-first + 选邻 接边，均限于 allowed
        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer_allowed(
                vec, current_node, l, self.ef_construction, allowed_ids)
            neighbours = self._select_neighbors(vec, candidates, self.M)
            for nb in neighbours:
                self._add_link(id, nb, l)
                self._add_link(nb, id, l)
            if neighbours:
                current_node = neighbours[0]

        if lsh is not None:
            lsh.add(id, vec)
        return id

    def add_item(self, vector: np.ndarray, id: Optional[int] = None) -> int:
        """Add a new vector (float32) to the index.
        Parameters
        ----------
        vector : np.ndarray
            Continuous vector.
        id : Optional[int], optional
            User-specified id; if None, an auto id is assigned.

        Returns
        -------
        int
            The id assigned to the inserted item.
        """
        vec = np.asarray(vector, dtype=np.float32)
        # assign id and level
        if id is None:
            id = self._id_counter
            self._id_counter += 1
        level = self._assign_level()
        self.items[id] = Node(vector=vec, level=level)

        # first element initializes the entry point
        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            return id

        # if new node has higher level than current max, set as new entry point
        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

        current_node = self.entry_point
        # insertion: search down from top level to level+1 greedily
        for l in range(self.max_level, level, -1):
            current_node = self._search_layer_greedy(vec, current_node, l)

        # search and connect neighbours on layers <= level
        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(
                vec, current_node, l, self.ef_construction)
            neighbours = self._select_neighbors(vec, candidates, self.M)
            for nb in neighbours:
                self._add_link(id, nb, l)
                self._add_link(nb, id, l)
            if neighbours:
                current_node = neighbours[0]
        return id

    def _add_link(self, source: int, dest: int, layer: int) -> None:
        """Add a directed link from ``source`` to ``dest`` on ``layer`` and prune."""
        nbrs = self.neighbours[layer][source]
        if dest not in nbrs:
            nbrs.append(dest)
            if len(nbrs) > self.M:
                # 计算到 source 的距离并保留最近的 M 个
                distances = [(n, self.distance(self.items[source].vector, self.items[n].vector))
                             for n in nbrs]
                distances.sort(key=lambda x: x[1])
                # 记录被删除的 cross distribution 边
                deleted_neighbors = [n for n, _ in distances[self.M:]]
                for deleted_nb in deleted_neighbors:
                    if self.edge_flags[layer][source].get(deleted_nb) == "cross_distribution":
                        self.cross_distribution_stats["deleted_cross_edges"] += 1
                        # 从标记中移除
                        if deleted_nb in self.edge_flags[layer][source]:
                            del self.edge_flags[layer][source][deleted_nb]

                self.neighbours[layer][source] = [
                    n for n, _ in distances[:self.M]]

    def _add_cross_distribution_link(self, source: int, dest: int, layer: int, query_id: Optional[int] = None) -> None:
        """Add a cross distribution link with marking."""
        nbrs = self.neighbours[layer][source]
        if dest not in nbrs:
            nbrs.append(dest)
            # 标记为 cross distribution 边
            self.edge_flags[layer][source][dest] = "cross_distribution"
            self.cross_distribution_stats["total_cross_edges"] += 1
            if query_id is not None:
                self.cross_distribution_stats["cross_edges_by_query"][query_id] += 1

            if len(nbrs) > self.M:
                # 计算到 source 的距离并保留最近的 M 个
                distances = [(n, self.distance(self.items[source].vector, self.items[n].vector))
                             for n in nbrs]
                distances.sort(key=lambda x: x[1])
                # 记录被删除的 cross distribution 边
                deleted_neighbors = [n for n, _ in distances[self.M:]]
                for deleted_nb in deleted_neighbors:
                    if self.edge_flags[layer][source].get(deleted_nb) == "cross_distribution":
                        self.cross_distribution_stats["deleted_cross_edges"] += 1
                        # 从标记中移除
                        if deleted_nb in self.edge_flags[layer][source]:
                            del self.edge_flags[layer][source][deleted_nb]

                self.neighbours[layer][source] = [
                    n for n, _ in distances[:self.M]]

    def _dist(self, u: int, v: int) -> float:
        """Pairwise distance between two data nodes by id.

        Assumes self.items[u].vector and self.items[v].vector are unit-normalized.
        Falls back to self.distance for custom metrics.
        """
        # missing id -> treat as infinitely far
        if u not in self.items or v not in self.items:
            return float("inf")

        a = self.items[u].vector
        b = self.items[v].vector

        # fast path for cosine_distance on unit-norm vectors
        try:
            # 如果你的 self.distance 指向上面的 cosine_distance 函数，
            # 直接使用 1 - dot 更快
            return 1.0 - float(a.dot(b))
        except Exception:
            # 兜底：使用通用的距离函数
            return float(self.distance(a, b))

    def _select_neighbors(self, vec: np.ndarray, candidates: Iterable[int], M: int) -> List[int]:
        """Pick the M closest candidate ids to vec (simple heuristic)."""
        cands = list(candidates)
        cands.sort(key=lambda cid: self.distance(vec, self.items[cid].vector))
        return cands[:M]

    def _search_layer_greedy(self, vec: np.ndarray, entry_id: int, layer: int) -> int:
        """Greedy search for the closest neighbour on a specific layer."""
        current = entry_id
        best_dist = self.distance(vec, self.items[current].vector)
        improved = True
        while improved:
            improved = False
            for nb in self.neighbours[layer][current]:
                dist = self.distance(vec, self.items[nb].vector)
                if dist < best_dist:
                    best_dist = dist
                    current = nb
                    improved = True
        return current

    # ===== Greedy search with trace (per layer) =====
    def _search_layer_greedy_trace(self, vec: np.ndarray, entry_id: int, layer: int):
        """Return (closest_id, steps, accel_edges); steps is a list of node ids, accel_edges is count of cross distribution edges used."""
        steps = []
        accel_edges = 0
        current = entry_id
        best_dist = self.distance(vec, self.items[current].vector)
        steps.append(current)
        improved = True
        while improved:
            improved = False
            for nb in self.neighbours[layer][current]:
                dist = self.distance(vec, self.items[nb].vector)
                steps.append(nb)

                # 检查是否经过 cross distribution 加速边
                if layer in self.edge_flags and current in self.edge_flags[layer]:
                    if nb in self.edge_flags[layer][current] and self.edge_flags[layer][current][nb] == "cross_distribution":
                        accel_edges += 1

                if dist < best_dist:
                    best_dist = dist
                    current = nb
                    improved = True
        return current, steps, accel_edges

    def _search_layer(self, vec: np.ndarray, entry_id: int, layer: int, ef: int) -> List[int]:
        """Best-first search within one layer (beam width = ef)."""
        import heapq
        visited: set[int] = set()
        candidates: List[Tuple[float, int]] = []   # min-heap by dist
        result: List[Tuple[float, int]] = []       # max-heap via negative dist

        dist_entry = self.distance(vec, self.items[entry_id].vector)
        heapq.heappush(candidates, (dist_entry, entry_id))
        heapq.heappush(result, (-dist_entry, entry_id))
        visited.add(entry_id)

        while candidates:
            dist_curr, curr = heapq.heappop(candidates)
            worst_dist = -result[0][0] if result else math.inf
            if dist_curr > worst_dist:
                break
            for nb in self.neighbours[layer][curr]:
                if nb in visited:
                    continue
                visited.add(nb)
                d = self.distance(vec, self.items[nb].vector)
                if len(result) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(result, (-d, nb))
                    if len(result) > ef:
                        heapq.heappop(result)
                        worst_dist = -result[0][0]
        return [nid for (_, nid) in result]

    # ===== Best-first search with trace (single layer, stop when target first seen) =====
    def _search_layer_trace_until_target(self, vec: np.ndarray, entry_id: int, layer: int, ef: int, target_id: int):
        import heapq
        steps = []
        visited: set[int] = set()
        candidates: List[Tuple[float, int]] = []   # min-heap by dist
        result: List[Tuple[float, int]] = []       # max-heap via negative dist

        dist_entry = self.distance(vec, self.items[entry_id].vector)
        heapq.heappush(candidates, (dist_entry, entry_id))
        heapq.heappush(result, (-dist_entry, entry_id))
        visited.add(entry_id)
        steps.append(entry_id)

        found = False
        while candidates:
            dist_curr, curr = heapq.heappop(candidates)
            worst_dist = -result[0][0] if result else math.inf

            if dist_curr > worst_dist:
                break

            for nb in self.neighbours[layer][curr]:
                if nb in visited:
                    continue
                visited.add(nb)
                d = self.distance(vec, self.items[nb].vector)
                steps.append(nb)
                if len(result) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(result, (-d, nb))
                    if len(result) > ef:
                        popped = heapq.heappop(result)
                        worst_dist = -result[0][0]
                if nb == target_id:
                    found = True
                    break
            if found:
                break
        return steps, found

    # ===== 新增：跨层搜索并检测加速边 =====

    def search_steps_to_target(self, vector: np.ndarray, target_id: int, k: int = 10, ef: Optional[int] = None,
                               verbose: bool = False, analyze_phases: bool = False,
                               enhanced_layer1_search: bool = True, layer1_ef: Optional[int] = None,
                               multi_path_search: bool = False, max_paths: int = 3):
        """Run a query across all layers and return the detailed navigation steps until the target is first reached.

        Parameters:
        -----------
        vector : np.ndarray
            Query vector
        target_id : int
            Target node ID to search for
        k : int
            Number of top-k results to return
        ef : Optional[int]
            Search beam width, if None uses self.ef_search
        verbose : bool
            Whether to print detailed distance information
        analyze_phases : bool
            Whether to analyze and print phase information
        enhanced_layer1_search : bool
            Whether to use enhanced search on layer 1 (default True)
        layer1_ef : Optional[int]
            Beam width for layer 1 search, if None uses ef*2
        multi_path_search : bool
            Whether to use multi-path concurrent search (default False)
        max_paths : int
            Maximum number of concurrent paths to explore (default 3)

        Returns:
        --------
        dict with keys:
            "found": bool,
            "trace": [ ... step dicts ... ],
            "final_candidates": List[Tuple[id, dist]]  # top-k at the end (optional)
            "phase_analysis": dict  # 如果 analyze_phases=True
        """
        if not self.items:
            return {"found": False, "trace": [], "final_candidates": []}

        vec = np.asarray(vector, dtype=np.float32)
        trace = []
        phase_analysis = {}

        # 1) 从最高层开始搜索，逐层向下，记录所有步骤
        curr = self.entry_point
        eff = max(self.ef_search, k) if ef is None else max(ef, k)
        layer1_eff = layer1_ef if layer1_ef is not None else eff * 2  # 第1层使用更大的beam width
        found = False

        # 在高层进行greedy搜索并记录步骤
        total_accel_edges = 0

        # 处理第2层及以上（如果有的话）
        for l in range(self.max_level, 1, -1):
            curr, steps, accel_edges = self._search_layer_greedy_trace(
                vec, curr, l)
            trace.extend(steps)
            total_accel_edges += accel_edges
            if verbose:
                print(
                    f"第{l}层搜索: 从 {steps[0] if steps else 'None'} 到 {curr}, 使用加速边: {accel_edges}")

        # 在第1层进行增强搜索（如果启用且存在第1层）
        if enhanced_layer1_search and self.max_level >= 1:
            if multi_path_search:
                if verbose:
                    print(
                        f"第1层多路并发搜索: 使用beam width {layer1_eff}, 最大路径数 {max_paths}")

                # 使用多路并发搜索
                steps, found, accel_edges, paths_explored = self._search_layer_multi_path(
                    vec, curr, 1, layer1_eff, target_id, max_paths, verbose
                )
                trace.extend(steps)
                total_accel_edges += accel_edges

                if verbose:
                    print(
                        f"第1层多路搜索完成: 步数 {len(steps)}, 使用加速边: {accel_edges}, 找到目标: {found}, 探索路径数: {paths_explored}")
            else:
                if verbose:
                    print(f"第1层增强搜索: 使用beam width {layer1_eff}")

                # 使用第1层增强搜索
                steps, found, accel_edges = self._search_layer1_enhanced(
                    vec, curr, 1, layer1_eff, target_id, verbose
                )
                trace.extend(steps)
                total_accel_edges += accel_edges

                if verbose:
                    print(
                        f"第1层搜索完成: 步数 {len(steps)}, 使用加速边: {accel_edges}, 找到目标: {found}")

            # 如果第1层找到了目标，直接返回
            if found:
                if analyze_phases:
                    search_type = "多路并发搜索" if multi_path_search else "增强搜索"
                    # 计算高层步数（总步数 - 第1层步数）
                    high_layer_steps = len(trace) - len(steps)

                    phase_analysis = {
                        "phase_1": {
                            "description": f"高层搜索+第1层{search_type}",
                            "steps": trace,  # 包含所有步数
                            "step_count": len(trace),
                            "accel_edges": total_accel_edges,
                            "accel_edge_ratio": total_accel_edges / len(trace) if trace else 0,
                            "nodes": [self.items[nid].vector for nid in trace]
                        },
                        "phase_2": {
                            "description": "未使用",
                            "steps": [],
                            "step_count": 0,
                            "accel_edges": 0,
                            "accel_edge_ratio": 0,
                            "nodes": []
                        },
                        "total_steps": len(trace),
                        "total_accel_edges": total_accel_edges,
                        "phase_1_ratio": 1.0,
                        "phase_2_ratio": 0.0,
                        "overall_accel_edge_ratio": total_accel_edges / len(trace) if trace else 0
                    }

                    # 如果是多路搜索，添加路径信息
                    if multi_path_search:
                        phase_analysis["paths_explored"] = paths_explored
                        phase_analysis["max_paths"] = max_paths

                # 获取最终候选结果
                cands = self._search_layer(vec, curr, 1, layer1_eff)
                dists = [(cid, self.distance(vec, self.items[cid].vector))
                         for cid in cands]
                dists.sort(key=lambda x: x[1])
                topk = dists[:k]

                result = {"found": found, "trace": trace,
                          "final_candidates": topk}
                if analyze_phases:
                    result["phase_analysis"] = phase_analysis
                return result
        else:
            # 如果第1层不存在或未启用增强搜索，使用原来的greedy搜索
            if self.max_level >= 1:
                curr, steps, accel_edges = self._search_layer_greedy_trace(
                    vec, curr, 1)
                trace.extend(steps)
                total_accel_edges += accel_edges
                if verbose:
                    print(
                        f"第1层greedy搜索: 从 {steps[0] if steps else 'None'} 到 {curr}, 使用加速边: {accel_edges}")

        # 在第0层进行详细搜索和阶段分析
        if analyze_phases:
            # 详细分析搜索阶段
            steps, found, phase_info = self._search_layer_trace_until_target_with_phases(
                vec, curr, 0, eff, target_id, verbose
            )
            trace.extend(steps)
            phase_analysis = phase_info

            # 将高层的加速边统计添加到阶段分析中
            if "phase_1" in phase_analysis:
                # 计算高层步数（总步数 - 第0层步数）
                layer0_steps = phase_analysis["phase_1"]["step_count"] + \
                    phase_analysis["phase_2"]["step_count"]
                high_layer_steps = len(trace) - layer0_steps

                # 将高层步数添加到第一阶段
                phase_analysis["phase_1"]["step_count"] += high_layer_steps
                phase_analysis["phase_1"]["steps"].extend(
                    trace[:high_layer_steps])

                # 高层的加速边应该被分配到第一阶段（快速靠近阶段）
                phase_analysis["phase_1"]["accel_edges"] += total_accel_edges
                phase_analysis["total_accel_edges"] += total_accel_edges
                phase_analysis["overall_accel_edge_ratio"] = phase_analysis["total_accel_edges"] / len(
                    trace) if trace else 0

                # 重新计算第一阶段的加速边比例
                if phase_analysis["phase_1"]["step_count"] > 0:
                    phase_analysis["phase_1"]["accel_edge_ratio"] = phase_analysis["phase_1"]["accel_edges"] / \
                        phase_analysis["phase_1"]["step_count"]

                # 更新总步数以包含所有层的步数
                phase_analysis["total_steps"] = len(trace)

                # 重新计算比例
                phase_analysis["phase_1_ratio"] = phase_analysis["phase_1"]["step_count"] / len(
                    trace) if trace else 0
                phase_analysis["phase_2_ratio"] = phase_analysis["phase_2"]["step_count"] / len(
                    trace) if trace else 0
        else:
            steps, found = self._search_layer_trace_until_target(
                vec, curr, 0, eff, target_id)
            trace.extend(steps)

        # 3) Optionally finish a normal search to return a final top-k list
        cands = self._search_layer(vec, curr, 0, eff)
        dists = [(cid, self.distance(vec, self.items[cid].vector))
                 for cid in cands]
        dists.sort(key=lambda x: x[1])
        topk = dists[:k]

        result = {"found": found, "trace": trace, "final_candidates": topk}
        if analyze_phases:
            result["phase_analysis"] = phase_analysis

        return result

    # ===== 新增：带阶段分析的搜索函数 =====
    def _search_layer_trace_until_target_with_phases(self, vec: np.ndarray, entry_id: int, layer: int,
                                                     ef: int, target_id: int, verbose: bool = False):
        """
        带阶段分析的搜索函数，区分快速靠近阶段和 beam search 阶段
        并统计每个阶段经过的 cross distribution 加速边
        """
        import heapq

        steps = []
        visited: set[int] = set()
        candidates: List[Tuple[float, int]] = []   # min-heap by dist
        result: List[Tuple[float, int]] = []       # max-heap via negative dist

        # 阶段分析相关变量
        phase_1_steps = []  # 第一阶段：top-1 持续变化
        phase_2_steps = []  # 第二阶段：top-1 相对固定
        current_top1 = entry_id
        top1_stable_count = 0
        top1_stable_threshold = 3  # top-1 连续保持不变的次数阈值

        # 加速边统计
        phase_1_accel_edges = 0  # 第一阶段经过的加速边数
        phase_2_accel_edges = 0  # 第二阶段经过的加速边数

        dist_entry = self.distance(vec, self.items[entry_id].vector)
        heapq.heappush(candidates, (dist_entry, entry_id))
        heapq.heappush(result, (-dist_entry, entry_id))
        visited.add(entry_id)
        steps.append(entry_id)

        # 将起始节点添加到第一阶段（初始状态）
        phase_1_steps.append(entry_id)

        if verbose:
            print(f"起始点: {entry_id}, 距离: {dist_entry:.6f}")

        found = False
        while candidates:
            dist_curr, curr = heapq.heappop(candidates)
            worst_dist = -result[0][0] if result else math.inf

            if dist_curr > worst_dist:
                break

            for nb in self.neighbours[layer][curr]:
                if nb in visited:
                    continue
                visited.add(nb)
                d = self.distance(vec, self.items[nb].vector)
                steps.append(nb)

                # 检查是否经过 cross distribution 加速边
                is_accel_edge = False
                if layer in self.edge_flags and curr in self.edge_flags[layer]:
                    if nb in self.edge_flags[layer][curr] and self.edge_flags[layer][curr][nb] == "cross_distribution":
                        is_accel_edge = True

                if verbose:
                    edge_type = "加速边" if is_accel_edge else "普通边"
                    print(
                        f"访问节点: {nb}, 距离: {d:.6f}, 当前最佳: {dist_curr:.6f}, 边类型: {edge_type}")

                if len(result) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(result, (-d, nb))
                    if len(result) > ef:
                        popped = heapq.heappop(result)
                        worst_dist = -result[0][0]

                # 检查是否找到目标
                if nb == target_id:
                    found = True
                    if verbose:
                        print(f"找到目标节点: {target_id}, 距离: {d:.6f}")
                    break

                # 阶段分析：检查 top-1 是否发生变化
                if result and result[0][1] != current_top1:
                    # top-1 发生变化
                    new_top1 = result[0][1]
                    new_top1_dist = -result[0][0]
                    if verbose:
                        print(
                            f"Top-1 变化: {current_top1} -> {new_top1}, 距离: {new_top1_dist:.6f}")

                    current_top1 = new_top1
                    top1_stable_count = 0
                else:
                    top1_stable_count += 1

                # 判断阶段并统计加速边
                if top1_stable_count < top1_stable_threshold:
                    # 第一阶段：top-1 持续变化
                    phase_1_steps.append(nb)
                    if is_accel_edge:
                        phase_1_accel_edges += 1
                else:
                    # 第二阶段：top-1 相对固定
                    phase_2_steps.append(nb)
                    if is_accel_edge:
                        phase_2_accel_edges += 1

            if found:
                break

        # 分析结果
        phase_analysis = {
            "phase_1": {
                "description": "快速靠近阶段 (top-1 持续变化)",
                "steps": phase_1_steps,
                "step_count": len(phase_1_steps),
                "accel_edges": phase_1_accel_edges,
                "accel_edge_ratio": phase_1_accel_edges / len(phase_1_steps) if phase_1_steps else 0,
                "nodes": [self.items[nid].vector for nid in phase_1_steps]
            },
            "phase_2": {
                "description": "Beam Search 阶段 (top-1 相对固定)",
                "steps": phase_2_steps,
                "step_count": len(phase_2_steps),
                "accel_edges": phase_2_accel_edges,
                "accel_edge_ratio": phase_2_accel_edges / len(phase_2_steps) if phase_2_steps else 0,
                "nodes": [self.items[nid].vector for nid in phase_2_steps]
            },
            "total_steps": len(steps),
            "total_accel_edges": phase_1_accel_edges + phase_2_accel_edges,
            "phase_1_ratio": len(phase_1_steps) / len(steps) if steps else 0,
            "phase_2_ratio": len(phase_2_steps) / len(steps) if steps else 0,
            "overall_accel_edge_ratio": (phase_1_accel_edges + phase_2_accel_edges) / len(steps) if steps else 0
        }

        if verbose:
            print(f"\n=== 阶段分析结果 ===")
            print(
                f"第一阶段 (快速靠近): {len(phase_1_steps)} 步, 占比: {phase_analysis['phase_1_ratio']:.2%}")
            print(
                f"  经过加速边: {phase_1_accel_edges} 条, 加速边比例: {phase_analysis['phase_1']['accel_edge_ratio']:.2%}")
            print(
                f"第二阶段 (Beam Search): {len(phase_2_steps)} 步, 占比: {phase_analysis['phase_2_ratio']:.2%}")
            print(
                f"  经过加速边: {phase_2_accel_edges} 条, 加速边比例: {phase_analysis['phase_2']['accel_edge_ratio']:.2%}")
            print(
                f"总步数: {len(steps)}, 总加速边: {phase_analysis['total_accel_edges']}, 整体加速边比例: {phase_analysis['overall_accel_edge_ratio']:.2%}")

            if phase_1_steps:
                print(
                    f"第一阶段节点: {phase_1_steps[:5]}{'...' if len(phase_1_steps) > 5 else ''}")
            if phase_2_steps:
                print(
                    f"第二阶段节点: {phase_2_steps[:5]}{'...' if len(phase_2_steps) > 5 else ''}")

        return steps, found, phase_analysis

    # ===== 新增：第1层增强搜索函数 =====
    def _search_layer1_enhanced(self, vec: np.ndarray, entry_id: int, layer: int, ef: int, target_id: int, verbose: bool = False):
        """
        第1层增强搜索函数，使用beam search并统计加速边
        返回 (steps, found, accel_edges)
        """
        import heapq

        steps = []
        visited: set[int] = set()
        candidates: List[Tuple[float, int]] = []   # min-heap by dist
        result: List[Tuple[float, int]] = []       # max-heap via negative dist

        # 加速边统计
        accel_edges = 0

        dist_entry = self.distance(vec, self.items[entry_id].vector)
        heapq.heappush(candidates, (dist_entry, entry_id))
        heapq.heappush(result, (-dist_entry, entry_id))
        visited.add(entry_id)
        steps.append(entry_id)

        if verbose:
            print(f"第1层增强搜索起始点: {entry_id}, 距离: {dist_entry:.6f}")

        found = False
        while candidates:
            dist_curr, curr = heapq.heappop(candidates)
            worst_dist = -result[0][0] if result else math.inf

            if dist_curr > worst_dist:
                break

            for nb in self.neighbours[layer][curr]:
                if nb in visited:
                    continue
                visited.add(nb)
                d = self.distance(vec, self.items[nb].vector)
                steps.append(nb)

                # 检查是否经过 cross distribution 加速边
                is_accel_edge = False
                if layer in self.edge_flags and curr in self.edge_flags[layer]:
                    if nb in self.edge_flags[layer][curr] and self.edge_flags[layer][curr][nb] == "cross_distribution":
                        is_accel_edge = True
                        accel_edges += 1

                if verbose:
                    edge_type = "加速边" if is_accel_edge else "普通边"
                    print(
                        f"第1层访问节点: {nb}, 距离: {d:.6f}, 当前最佳: {dist_curr:.6f}, 边类型: {edge_type}")

                if len(result) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(result, (-d, nb))
                    if len(result) > ef:
                        popped = heapq.heappop(result)
                        worst_dist = -result[0][0]

                # 检查是否找到目标
                if nb == target_id:
                    found = True
                    if verbose:
                        print(f"第1层找到目标节点: {target_id}, 距离: {d:.6f}")
                    break

            if found:
                break

        if verbose:
            print(
                f"第1层增强搜索完成: 步数 {len(steps)}, 加速边 {accel_edges}, 找到目标: {found}")

        return steps, found, accel_edges

    # ===== 新增：多路并发搜索函数 =====
    def _search_layer_multi_path(self, vec: np.ndarray, entry_id: int, layer: int, ef: int,
                                 target_id: int, max_paths: int = 3, verbose: bool = False):
        """
        多路并发搜索函数，同时探索多条路径
        返回 (steps, found, accel_edges, paths_explored)
        """
        import heapq

        steps = []
        visited: set[int] = set()
        candidates: List[Tuple[float, int]] = []   # min-heap by dist
        result: List[Tuple[float, int]] = []       # max-heap via negative dist

        # 多路并发相关变量
        active_paths = []  # 当前活跃的路径
        paths_explored = 0
        accel_edges = 0

        # 初始化第一条路径
        dist_entry = self.distance(vec, self.items[entry_id].vector)
        heapq.heappush(candidates, (dist_entry, entry_id))
        heapq.heappush(result, (-dist_entry, entry_id))
        visited.add(entry_id)
        steps.append(entry_id)
        active_paths.append([entry_id])
        paths_explored = 1

        if verbose:
            print(
                f"多路并发搜索起始点: {entry_id}, 距离: {dist_entry:.6f}, 最大路径数: {max_paths}")

        found = False
        while candidates and not found:
            # 获取当前最佳候选
            dist_curr, curr = heapq.heappop(candidates)
            worst_dist = -result[0][0] if result else math.inf

            if dist_curr > worst_dist:
                break

            # 探索当前节点的所有邻居
            neighbors_to_explore = []
            for nb in self.neighbours[layer][curr]:
                if nb in visited:
                    continue
                visited.add(nb)
                d = self.distance(vec, self.items[nb].vector)
                neighbors_to_explore.append((d, nb))

            if not neighbors_to_explore:
                continue

            # 按距离排序邻居
            neighbors_to_explore.sort(key=lambda x: x[0])

            # 多路并发：同时探索多个邻居
            paths_to_add = []
            for i, (d, nb) in enumerate(neighbors_to_explore):
                if i >= max_paths:
                    break  # 限制并发路径数

                steps.append(nb)

                # 检查是否经过 cross distribution 加速边
                is_accel_edge = False
                if layer in self.edge_flags and curr in self.edge_flags[layer]:
                    if nb in self.edge_flags[layer][curr] and self.edge_flags[layer][curr][nb] == "cross_distribution":
                        is_accel_edge = True
                        accel_edges += 1

                if verbose:
                    edge_type = "加速边" if is_accel_edge else "普通边"
                    print(
                        f"多路访问节点: {nb}, 距离: {d:.6f}, 路径: {i+1}, 边类型: {edge_type}")

                # 添加到候选队列
                if len(result) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(result, (-d, nb))
                    if len(result) > ef:
                        popped = heapq.heappop(result)
                        worst_dist = -result[0][0]

                    # 记录新路径
                    paths_to_add.append(nb)

                # 检查是否找到目标
                if nb == target_id:
                    found = True
                    if verbose:
                        print(
                            f"多路搜索找到目标节点: {target_id}, 距离: {d:.6f}, 路径: {i+1}")
                    break

            # 更新活跃路径
            if paths_to_add:
                new_active_paths = []
                for path in active_paths:
                    for nb in paths_to_add:
                        new_path = path + [nb]
                        new_active_paths.append(new_path)
                        paths_explored += 1

                active_paths = new_active_paths[:max_paths]  # 限制路径数量

            if found:
                break

        if verbose:
            print(
                f"多路并发搜索完成: 步数 {len(steps)}, 加速边 {accel_edges}, 找到目标: {found}, 探索路径数: {paths_explored}")

        return steps, found, accel_edges, paths_explored

    # ====== 基于Query的Cross Distribution边构建 ======
    def build_cross_distribution_edges(
        self,
        query: np.ndarray,
        top_k: int = 10,
        max_new_edges_per_node: int = 4,
        random_seed: int = None,
    ) -> dict:
        """
        基于query在layer 1中构建cross distribution边：
        1. 根据query检索到layer 1
        2. 在layer 1内遍历全部节点，找到最后的top10
        3. 在top10内，如果任意两个节点之间没有边，就新增一条边

        参数:
        ----------
        query : np.ndarray
            查询向量
        top_k : int
            在layer 1中检索的top-k节点数量，默认10
        max_new_edges_per_node : int
            每个节点最多新增的边数
        random_seed : int
            随机种子
        """
        import random

        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)

        # 检查是否有layer 1
        if self.max_level < 1 or 1 not in self.neighbours:
            return {"error": "Layer 1 not available"}

        # 处理查询向量
        query_vec = np.asarray(query, dtype=np.float32)

        # 每个节点本次新增计数（限流）
        from collections import defaultdict
        added_per_node: Dict[int, int] = defaultdict(int)

        stats = {
            "query_processed": True,
            "layer_1_nodes_total": 0,
            "top_k_selected": 0,
            "pairs_considered": 0,
            "pairs_added": 0,
            "skipped_existing": 0,
            "pruned_by_cap": 0,
            "edges_added": 0,
            "top_k_nodes": [],
            "query_distance": 0.0
        }

        def can_add(u: int, v: int, current_layer: int):
            # 度预算
            if (max_new_edges_per_node is not None) and ((added_per_node[u] >= max_new_edges_per_node) or (added_per_node[v] >= max_new_edges_per_node)):
                stats["pruned_by_cap"] += 1
                return False
            # 检查是否已存在边
            if v in self.neighbours[current_layer][u]:
                stats["skipped_existing"] += 1
                return False
            return True

        def _add_cross_pair(u: int, v: int, current_layer: int):
            # 双向加 cross distribution 边
            self._add_cross_distribution_link(
                u, v, current_layer, None)  # query_id设为None
            self._add_cross_distribution_link(v, u, current_layer, None)
            added_per_node[u] += 1
            added_per_node[v] += 1
            stats["pairs_added"] += 1
            stats["edges_added"] += 1

        # 获取layer 1中的所有节点
        layer_1_nodes = list(self.neighbours[1].keys())
        stats["layer_1_nodes_total"] = len(layer_1_nodes)

        if len(layer_1_nodes) < 2:
            return {"error": "Not enough nodes in layer 1"}

        # 计算query到layer 1中所有节点的距离
        node_distances = []
        for node_id in layer_1_nodes:
            if node_id in self.items:
                dist = self.distance(query_vec, self.items[node_id].vector)
                node_distances.append((dist, node_id))

        # 按距离排序，选择top-k
        node_distances.sort(key=lambda x: x[0])
        top_k_nodes = [
            node_id for _, node_id in node_distances[:min(top_k, len(node_distances))]]
        stats["top_k_selected"] = len(top_k_nodes)
        stats["top_k_nodes"] = top_k_nodes

        if len(top_k_nodes) < 2:
            return {"error": "Not enough nodes in top-k selection"}

        # 记录query到最近节点的距离
        if node_distances:
            stats["query_distance"] = node_distances[0][0]

        # 在top-k节点中，检查任意两个节点之间是否有边，如果没有就添加
        for i in range(len(top_k_nodes)):
            for j in range(i + 1, len(top_k_nodes)):
                u, v = top_k_nodes[i], top_k_nodes[j]
                stats["pairs_considered"] += 1

                # 检查是否已经有边，如果没有就添加
                if can_add(u, v, 1):  # 在layer 1中添加边
                    _add_cross_pair(u, v, 1)

        return stats

    def get_cross_distribution_stats(self) -> dict:
        """获取 cross distribution 边的统计信息"""
        return {
            "total_cross_edges": self.cross_distribution_stats["total_cross_edges"],
            "deleted_cross_edges": self.cross_distribution_stats["deleted_cross_edges"],
            "active_cross_edges": self.cross_distribution_stats["total_cross_edges"] - self.cross_distribution_stats["deleted_cross_edges"],
            "cross_edges_by_query": dict(self.cross_distribution_stats["cross_edges_by_query"])
        }

    def reset_cross_distribution_stats(self) -> None:
        """重置 cross distribution 统计信息"""
        self.cross_distribution_stats = {
            "total_cross_edges": 0,
            "deleted_cross_edges": 0,
            "cross_edges_by_query": defaultdict(int)
        }

    def query(self, vector: np.ndarray, k: int = 10, ef: Optional[int] = None,
              track_steps: bool = False) -> List[int]:
        """
        使用HNSW索引进行查询

        Parameters:
        -----------
        vector : np.ndarray
            查询向量
        k : int
            返回top-k结果
        ef : Optional[int]
            搜索beam width，如果None使用self.ef_search
        track_steps : bool
            是否跟踪搜索步数

        Returns:
        --------
        List[int]
            返回的节点ID列表
        """
        if not self.items:
            return []

        vec = np.asarray(vector, dtype=np.float32)
        eff = max(self.ef_search, k) if ef is None else max(ef, k)

        # 从最高层开始搜索
        current_node = self.entry_point
        for l in range(self.max_level, 0, -1):
            current_node = self._search_layer_greedy(vec, current_node, l)

        # 在第0层进行beam search
        candidates = self._search_layer(vec, current_node, 0, eff)

        # 计算距离并排序
        dists = [(cid, self.distance(vec, self.items[cid].vector))
                 for cid in candidates]
        dists.sort(key=lambda x: x[1])

        return [cid for cid, _ in dists[:k]]

    def query_with_steps(self, vector: np.ndarray, k: int = 10, ef: Optional[int] = None) -> Tuple[List[int], int]:
        """
        使用HNSW索引进行查询并跟踪搜索步数

        Parameters:
        -----------
        vector : np.ndarray
            查询向量
        k : int
            返回top-k结果
        ef : Optional[int]
            搜索beam width，如果None使用self.ef_search

        Returns:
        --------
        Tuple[List[int], int]
            (返回的节点ID列表, 搜索步数)
        """
        if not self.items:
            return [], 0

        vec = np.asarray(vector, dtype=np.float32)
        eff = max(self.ef_search, k) if ef is None else max(ef, k)

        total_steps = 0

        # 从最高层开始搜索，跟踪步数
        current_node = self.entry_point
        for l in range(self.max_level, 0, -1):
            current_node, steps, _ = self._search_layer_greedy_trace(
                vec, current_node, l)
            total_steps += len(steps)

        # 在第0层进行beam search，跟踪步数
        candidates = self._search_layer(vec, current_node, 0, eff)
        total_steps += len(candidates)  # 近似步数

        # 计算距离并排序
        dists = [(cid, self.distance(vec, self.items[cid].vector))
                 for cid in candidates]
        dists.sort(key=lambda x: x[1])

        return [cid for cid, _ in dists[:k]], total_steps
