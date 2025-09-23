# from __future__ import annotations

"""
HNSW Index with Layer-by-Layer Building

This file is a modified version of hnsw_cosine_status_high.py with the following key changes:
1. build_cross_distribution_edges() now implements layer-by-layer building instead of random connections
2. First builds layer 0 with all nodes
3. Then randomly selects 5% of nodes from layer 0 to build layer 1
4. Then randomly selects 5% of nodes from layer 1 to build layer 2
5. Total of 3 layers (0, 1, 2)

The new approach ensures a hierarchical structure where higher layers are built from lower layers,
creating a more structured and predictable graph topology.
"""

import math
import random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

from simple_sim_hash import SimpleSimHash

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 先离线把向量做 unit-norm，再使用 1 - 内积（= 1 - cos）
    return float(1.0 - np.dot(a, b))

def cosine_distance_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    vec: shape (d,)
    arr: shape (n, d)
    要求 vec 和 arr 已经预先 unit-norm 归一化
    """
    # 一次性计算所有 dot product
    a = a.reshape(1, 200)
    b = b.reshape(-1,200)
    dots = np.matmul(a, b.T)  # shape (n,)
    
    # cosine distance = 1 - cos
    distances = 1.0 - dots
    return distances.flatten()

def _unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)

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
    """Hierarchical Navigable Small World index using cosine distance.

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
        Distance function; default is cosine_distance.
    """

    def __init__(
        self,
        M: int = 8,
        ef_construction: int = 200,
        ef_search: int = 50,
        random_seed: Optional[int] = None,
        distance_fn=cosine_distance,
    ) -> None:
        self.M = M
        self.ef_construction = max(ef_construction, M)
        self.ef_search = max(ef_search, M)
        self.distance = distance_fn

        # map node id -> Node
        self.items: Dict[int, Node] = {}
        # adjacency lists per layer: layer -> id -> list of neighbour ids
        self.neighbours: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        # 新增：边的标记信息，用于跟踪 cross distribution 边
        self.edge_flags: Dict[int, Dict[int, Dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
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
                if nid in visited: continue
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
                    d = cosine_distance_batch(vec, np.array(item_pool))
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
            dists = cosine_distance_batch(vec, np.array(item_nbs))
            j = int(np.argmin(dists))
            if dists[j] < best_dist:
                best_dist = float(dists[j])
                current = nbs[j]
                improved = True
        return current

    def _search_layer_allowed(self, vec: np.ndarray, entry_id: int, layer: int, ef: int,
                            allowed: Optional[set] = None) -> List[int]:
        import heapq, math
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
            dvec = cosine_distance_batch(vec, np.array(item_nbs))
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
        vec = (vector / (np.linalg.norm(vector) + 1e-12)).astype(np.float32, copy=False)
        if id is None:
            id = self._id_counter
            self._id_counter += 1
        level = self._assign_level()
        self.items[id] = Node(vector=vec, level=level)

        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            if lsh is not None: lsh.add(id, vec)
            return id

        if level > self.max_level:
            self.max_level = level
            self.entry_point = id

        # 生成候选池（最多1万）
        allowed_ids = set(self.select_candidate_ids(vec, limit=limit, lsh=lsh))

        # 自顶向下：greedy 限于 allowed
        current_node = self.entry_point
        for l in range(self.max_level, level, -1):
            current_node = self._search_layer_greedy_allowed(vec, current_node, l, allowed_ids)

        # 底层及以下各层：单层 best-first + 选邻 接边，均限于 allowed
        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer_allowed(vec, current_node, l, self.ef_construction, allowed_ids)
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
        vec = _unit_norm(np.asarray(vector, dtype=np.float32))
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
            candidates = self._search_layer(vec, current_node, l, self.ef_construction)
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
                
                self.neighbours[layer][source] = [n for n, _ in distances[:self.M]]

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
                
                self.neighbours[layer][source] = [n for n, _ in distances[:self.M]]


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
                    print(f"第1层访问节点: {nb}, 距离: {d:.6f}, 当前最佳: {dist_curr:.6f}, 边类型: {edge_type}")
                
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
            print(f"第1层增强搜索完成: 步数 {len(steps)}, 加速边 {accel_edges}, 找到目标: {found}")
        
        return steps, found, accel_edges

    # ===== 新增：跨层搜索并检测加速边 =====
    def search_steps_to_target(self, vector: np.ndarray, target_id: int, k: int = 10, ef: Optional[int] = None, 
                                          verbose: bool = False, analyze_phases: bool = False,
                                          enhanced_layer1_search: bool = True, layer1_ef: Optional[int] = None):
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

        vec = _unit_norm(np.asarray(vector, dtype=np.float32))
        trace = []
        phase_analysis = {}

        # 1) 从最高层开始搜索，逐层向下，记录所有步骤
        curr = self.entry_point
        eff = max(self.ef_search, k) if ef is None else max(ef, k)
        found = False
        
        # 在高层进行greedy搜索并记录步骤
        total_accel_edges = 0
        for l in range(self.max_level, 1, -1):  # 从最高层到第2层
            # 检查该层是否存在
            if l in self.neighbours and l in self.edge_flags:
                curr, steps, accel_edges = self._search_layer_greedy_trace(vec, curr, l)
                trace.extend(steps)
                total_accel_edges += accel_edges
                if verbose:
                    print(f"第{l}层搜索: 从 {steps[0] if steps else 'None'} 到 {curr}, 使用加速边: {accel_edges}")
            else:
                if verbose:
                    print(f"第{l}层不存在，跳过")

        # 第1层增强搜索
        if self.max_level >= 1 and 1 in self.neighbours and 1 in self.edge_flags:
            if enhanced_layer1_search:
                # 使用增强的第1层搜索
                layer1_beam_width = layer1_ef if layer1_ef is not None else eff * 2
                steps, found_in_layer1, accel_edges = self._search_layer1_enhanced(
                    vec, curr, 1, layer1_beam_width, target_id, verbose
                )
                trace.extend(steps)
                total_accel_edges += accel_edges
                if verbose:
                    print(f"第1层增强搜索: 步数 {len(steps)}, 加速边 {accel_edges}, 找到目标: {found_in_layer1}")
                
                if found_in_layer1:
                    found = True
                    if verbose:
                        print(f"在第1层找到目标，提前终止搜索")
                    # 即使在第1层找到目标，也需要阶段分析信息
                    if analyze_phases:
                        # 为第1层搜索创建阶段分析信息
                        phase_analysis = {
                            "phase_1": {
                                "description": "第1层增强搜索",
                                "steps": trace,  # 使用完整的trace
                                "step_count": len(trace),
                                "accel_edges": total_accel_edges,
                                "accel_edge_ratio": total_accel_edges / len(trace) if trace else 0,
                                "nodes": [self.items[nid].vector for nid in trace]
                            },
                            "phase_2": {
                                "description": "第1层增强搜索",
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
                else:
                    # 如果第1层没找到，继续到第0层
                    curr = steps[-1] if steps else curr
            else:
                # 使用传统的greedy搜索
                curr, steps, accel_edges = self._search_layer_greedy_trace(vec, curr, 1)
                trace.extend(steps)
                total_accel_edges += accel_edges
                if verbose:
                    print(f"第1层greedy搜索: 从 {steps[0] if steps else 'None'} 到 {curr}, 使用加速边: {accel_edges}")
                
                # 为传统greedy搜索也创建阶段分析信息（如果需要）
                if analyze_phases and not found:
                    phase_analysis = {
                        "phase_1": {
                            "description": "第1层greedy搜索",
                            "steps": trace,  # 使用完整的trace
                            "step_count": len(trace),
                            "accel_edges": total_accel_edges,
                            "accel_edge_ratio": total_accel_edges / len(trace) if trace else 0,
                            "nodes": [self.items[nid].vector for nid in trace]
                        },
                        "phase_2": {
                            "description": "第1层greedy搜索",
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
        else:
            if verbose:
                print(f"第1层不存在，跳过")

        # 在第0层进行详细搜索和阶段分析（如果还没找到目标）
        if not found:
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
                    layer0_steps = phase_analysis["phase_1"]["step_count"] + phase_analysis["phase_2"]["step_count"]
                    high_layer_steps = len(trace) - layer0_steps
                    
                    # 将高层步数添加到第一阶段
                    phase_analysis["phase_1"]["step_count"] += high_layer_steps
                    phase_analysis["phase_1"]["steps"].extend(trace[:high_layer_steps])
                    
                    # 高层的加速边应该被分配到第一阶段（快速靠近阶段）
                    phase_analysis["phase_1"]["accel_edges"] += total_accel_edges
                    phase_analysis["total_accel_edges"] += total_accel_edges
                    phase_analysis["overall_accel_edge_ratio"] = phase_analysis["total_accel_edges"] / len(trace) if trace else 0
                    
                    # 重新计算第一阶段的加速边比例
                    if phase_analysis["phase_1"]["step_count"] > 0:
                        phase_analysis["phase_1"]["accel_edge_ratio"] = phase_analysis["phase_1"]["accel_edges"] / phase_analysis["phase_1"]["step_count"]
                    
                    # 更新总步数以包含所有层的步数
                    phase_analysis["total_steps"] = len(trace)
                    
                    # 重新计算比例
                    phase_analysis["phase_1_ratio"] = phase_analysis["phase_1"]["step_count"] / len(trace) if trace else 0
                    phase_analysis["phase_2_ratio"] = phase_analysis["phase_2"]["step_count"] / len(trace) if trace else 0
            else:
                steps, found = self._search_layer_trace_until_target(vec, curr, 0, eff, target_id)
                trace.extend(steps)

        # 3) Optionally finish a normal search to return a final top-k list
        cands = self._search_layer(vec, curr, 0, eff)
        dists = [(cid, self.distance(vec, self.items[cid].vector)) for cid in cands]
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
        
        # 将起始节点分配到第一阶段
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
                    print(f"访问节点: {nb}, 距离: {d:.6f}, 当前最佳: {dist_curr:.6f}, 边类型: {edge_type}")
                
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
                        print(f"Top-1 变化: {current_top1} -> {new_top1}, 距离: {new_top1_dist:.6f}")
                    
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
            print(f"第一阶段 (快速靠近): {len(phase_1_steps)} 步, 占比: {phase_analysis['phase_1_ratio']:.2%}")
            print(f"  经过加速边: {phase_1_accel_edges} 条, 加速边比例: {phase_analysis['phase_1']['accel_edge_ratio']:.2%}")
            print(f"第二阶段 (Beam Search): {len(phase_2_steps)} 步, 占比: {phase_analysis['phase_2_ratio']:.2%}")
            print(f"  经过加速边: {phase_2_accel_edges} 条, 加速边比例: {phase_analysis['phase_2']['accel_edge_ratio']:.2%}")
            print(f"总步数: {len(steps)}, 总加速边: {phase_analysis['total_accel_edges']}, 整体加速边比例: {phase_analysis['overall_accel_edge_ratio']:.2%}")
            
            if phase_1_steps:
                print(f"第一阶段节点: {phase_1_steps[:5]}{'...' if len(phase_1_steps) > 5 else ''}")
            if phase_2_steps:
                print(f"第二阶段节点: {phase_2_steps[:5]}{'...' if len(phase_2_steps) > 5 else ''}")
        
        return steps, found, phase_analysis

    # ====== 删除高层并重建的构建逻辑 ======
    def build_cross_distribution_edges(
        self,
        max_new_edges_per_node: int = 4,
        random_seed: int = None,
        degree_threshold_ratio: float = 0.5,  # 出度阈值比例，默认50%
        max_layers: int = 3,  # 最大层数，包括第0层
        cross_dist_weight: float = 2.0,  # cross distribution边的权重倍数
        prioritize_cross_dist: bool = True,  # 是否优先选择有cross distribution边的节点
    ) -> dict:
        """
        删除第一层及以上的所有层，并从第0层向上生长重建：
        1. 删除第一层及以上的所有层，但记录每层的节点个数
        2. 从第0层向上重建，优先选择有cross distribution边的节点
        3. 保证每个节点的neighbor没有被选中
        4. 按照标准HNSW创建每一层
        5. 保持层数不变，每层的节点数不变
        6. 修改entry_point随机选择最高层的一个节点
        
        参数:
        ----------
        max_new_edges_per_node : int
            每个节点最多新增的边数
        random_seed : int
            随机种子
        degree_threshold_ratio : float
            出度阈值比例，默认50%
        max_layers : int
            最大层数，包括第0层，默认3层
        cross_dist_weight : float
            cross distribution边的权重倍数，默认2.0
        prioritize_cross_dist : bool
            是否优先选择有cross distribution边的节点，默认True
        """
        import random
        
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)
        
        # 检查是否有足够的节点
        if len(self.items) < 2:
            return {"error": "Not enough nodes to build layers"}
        
        # 每个节点本次新增计数（限流）
        from collections import defaultdict
        added_per_node: Dict[int, int] = defaultdict(int)

        stats = {
            "total_nodes": len(self.items),
            "layers_built": 0,
            "nodes_by_layer": defaultdict(int),
            "edges_by_layer": defaultdict(int),
            "degree_threshold_ratio": degree_threshold_ratio,
            "max_layers": max_layers,
            "cross_dist_weight": cross_dist_weight,
            "prioritize_cross_dist": prioritize_cross_dist,
            "selection_details": defaultdict(dict)
        }

        def can_add(u: int, v: int, current_layer: int):
            # 度预算
            if (max_new_edges_per_node is not None) and ( (added_per_node[u] >= max_new_edges_per_node) or (added_per_node[v] >= max_new_edges_per_node) ):
                return False
            # 检查是否已存在边
            if v in self.neighbours[current_layer][u]:
                return False
            return True

        def _add_cross_pair(u: int, v: int, current_layer: int):
            # 双向加 cross distribution 边
            self._add_cross_distribution_link(u, v, current_layer, None)  # query_id设为None
            self._add_cross_distribution_link(v, u, current_layer, None)
            added_per_node[u] += 1
            added_per_node[v] += 1
            stats["edges_by_layer"][current_layer] += 1

        # 第一步：记录现有各层的节点分布
        existing_layers = {}
        for node_id, node in self.items.items():
            level = node.level
            if level not in existing_layers:
                existing_layers[level] = []
            existing_layers[level].append(node_id)
        
        # 记录每层的节点个数
        layer_node_counts = {}
        for level, nodes in existing_layers.items():
            layer_node_counts[level] = len(nodes)
        
        print(f"记录现有层分布: {layer_node_counts}")
        
        # 第二步：删除第一层及以上的所有层
        print("删除第一层及以上的所有层...")
        layers_to_delete = [level for level in existing_layers.keys() if level >= 1]
        
        for layer in layers_to_delete:
            # 删除该层的所有邻居关系
            if layer in self.neighbours:
                del self.neighbours[layer]
            # 删除该层的边标记
            if layer in self.edge_flags:
                del self.edge_flags[layer]
            # 将该层的所有节点level重置为0
            for node_id in existing_layers[layer]:
                self.items[node_id].level = 0
        
        # 确保第0层存在
        if 0 not in self.neighbours:
            self.neighbours[0] = defaultdict(list)
        if 0 not in self.edge_flags:
            self.edge_flags[0] = defaultdict(dict)
        
        print(f"删除的层: {layers_to_delete}")
        
        # 记录第0层信息
        layer_0_nodes = list(self.items.keys())
        stats["nodes_by_layer"][0] = len(layer_0_nodes)
        
        # 统计第0层中的cross distribution节点
        layer_0_cross_dist_nodes = 0
        layer_0_cross_dist_edges = 0
        for node_id in layer_0_nodes:
            if 0 in self.edge_flags and node_id in self.edge_flags[0]:
                has_cross_dist = False
                for neighbor_id in self.neighbours[0].get(node_id, []):
                    if neighbor_id in self.edge_flags[0][node_id]:
                        if self.edge_flags[0][node_id][neighbor_id] == "cross_distribution":
                            has_cross_dist = True
                            layer_0_cross_dist_edges += 1
                if has_cross_dist:
                    layer_0_cross_dist_nodes += 1
        
        stats["selection_details"][0] = {
            "total_nodes": len(layer_0_nodes),
            "selected_nodes": len(layer_0_nodes),
            "selection_ratio": 1.0,
            "selected_cross_dist_nodes": layer_0_cross_dist_nodes,
            "cross_dist_nodes_ratio": layer_0_cross_dist_nodes / len(layer_0_nodes) if layer_0_nodes else 0,
            "total_cross_dist_edges_in_selected": layer_0_cross_dist_edges,
            "note": "base_layer"
        }
        stats["layers_built"] = 1
        
        # 第三步：从第0层向上生长，按标准HNSW构建新层
        current_layer_nodes = layer_0_nodes.copy()
        
        for target_layer in range(1, max_layers):
            # 检查是否需要重建这一层
            if target_layer not in layer_node_counts:
                break  # 如果原层不存在，停止构建
            
            target_node_count = layer_node_counts[target_layer]
            if target_node_count < 1:
                break  # 如果原层没有节点，停止构建
            
            # 计算出度阈值
            degree_threshold = max(1, int(degree_threshold_ratio * self.M))
            
            # 计算每个节点的综合评分（优先考虑cross distribution边）
            node_scores = []
            for node_id in current_layer_nodes:
                # 基础出度
                degree = len(self.neighbours[0].get(node_id, []))
                
                # 计算cross distribution边数量
                cross_dist_edges = 0
                if 0 in self.edge_flags and node_id in self.edge_flags[0]:
                    for neighbor_id in self.neighbours[0].get(node_id, []):
                        if neighbor_id in self.edge_flags[0][node_id]:
                            if self.edge_flags[0][node_id][neighbor_id] == "cross_distribution":
                                cross_dist_edges += 1
                
                # 综合评分：cross distribution边权重更高
                # 基础出度权重为1，cross distribution边权重可配置
                total_score = degree + cross_dist_edges * cross_dist_weight
                
                node_scores.append((node_id, degree, cross_dist_edges, total_score))
            
            # 按综合评分排序，优先选择有cross distribution边的节点
            node_scores.sort(key=lambda x: (x[2], x[3]), reverse=True)  # 先按cross_dist_edges，再按total_score
            
            # 选择节点，优先选择有cross distribution边的节点
            selected_nodes = []
            selected_set = set()
            
            # 第一轮：优先选择有cross distribution边的节点（如果启用）
            if prioritize_cross_dist:
                for node_id, degree, cross_dist_edges, total_score in node_scores:
                    if len(selected_nodes) >= target_node_count:
                        break
                    
                    # 检查该节点的neighbor是否已被选中
                    neighbors = self.neighbours[0].get(node_id, [])
                    neighbor_conflict = any(nb in selected_set for nb in neighbors)
                    
                    # 优先选择有cross distribution边的节点
                    if cross_dist_edges > 0 and not neighbor_conflict:
                        selected_nodes.append(node_id)
                        selected_set.add(node_id)
            
            # 第二轮：如果还不够，选择出度高的节点
            if len(selected_nodes) < target_node_count:
                for node_id, degree, cross_dist_edges, total_score in node_scores:
                    if len(selected_nodes) >= target_node_count:
                        break
                    
                    if node_id in selected_set:
                        continue
                    
                    # 检查该节点的neighbor是否已被选中
                    neighbors = self.neighbours[0].get(node_id, [])
                    neighbor_conflict = any(nb in selected_set for nb in neighbors)
                    
                    # 如果出度满足阈值且没有neighbor冲突，则选中
                    if degree >= degree_threshold and not neighbor_conflict:
                        selected_nodes.append(node_id)
                        selected_set.add(node_id)
            
            # 第三轮：如果还不够，从剩余节点中随机选择
            if len(selected_nodes) < target_node_count:
                remaining_nodes = [node_id for node_id in current_layer_nodes if node_id not in selected_set]
                needed = target_node_count - len(selected_nodes)
                additional_nodes = random.sample(remaining_nodes, min(needed, len(remaining_nodes)))
                selected_nodes.extend(additional_nodes)
                selected_set.update(additional_nodes)
            
            
            # 统计选中的节点中有多少具有cross distribution边
            selected_cross_dist_nodes = 0
            total_cross_dist_edges_in_selected = 0
            for node_id in selected_nodes:
                if 0 in self.edge_flags and node_id in self.edge_flags[0]:
                    has_cross_dist = False
                    for neighbor_id in self.neighbours[0].get(node_id, []):
                        if neighbor_id in self.edge_flags[0][node_id]:
                            if self.edge_flags[0][node_id][neighbor_id] == "cross_distribution":
                                has_cross_dist = True
                                total_cross_dist_edges_in_selected += 1
                    if has_cross_dist:
                        selected_cross_dist_nodes += 1
            
            # 记录选择详情
            stats["selection_details"][target_layer] = {
                "total_nodes": len(current_layer_nodes),
                "selected_nodes": len(selected_nodes),
                "target_count": target_node_count,
                "degree_threshold": degree_threshold,
                "selection_ratio": len(selected_nodes) / len(current_layer_nodes),
                "selected_cross_dist_nodes": selected_cross_dist_nodes,
                "cross_dist_nodes_ratio": selected_cross_dist_nodes / len(selected_nodes) if selected_nodes else 0,
                "total_cross_dist_edges_in_selected": total_cross_dist_edges_in_selected,
                "note": "rebuilt_layer_with_cross_distribution_priority"
            }
            
            # 为选中的节点更新level并构建当前层
            layer_nodes = []
            for node_id in selected_nodes:
                # 更新节点的level
                self.items[node_id].level = target_layer
                # 确保节点在当前层有邻居列表
                if node_id not in self.neighbours[target_layer]:
                    self.neighbours[target_layer][node_id] = []
                layer_nodes.append(node_id)
            
            stats["nodes_by_layer"][target_layer] = len(layer_nodes)
            
            # 按标准HNSW方式构建当前层的连接
            if len(layer_nodes) >= 2:
                # 为每个选中的节点构建连接
                for node_id in layer_nodes:
                    node_vec = self.items[node_id].vector
                    
                    # 在当前层中找到所有其他节点
                    other_nodes = [nid for nid in layer_nodes if nid != node_id]
                    if not other_nodes:
                        continue
                    
                    # 计算到其他节点的距离
                    distances = []
                    for other_node_id in other_nodes:
                        dist = self.distance(node_vec, self.items[other_node_id].vector)
                        distances.append((dist, other_node_id))
                    
                    # 按距离排序，选择最近的M个邻居
                    distances.sort(key=lambda x: x[0])
                    neighbors_to_add = distances[:self.M]
                    
                    # 添加邻居连接
                    for dist, neighbor_id in neighbors_to_add:
                        if can_add(node_id, neighbor_id, target_layer):
                            _add_cross_pair(node_id, neighbor_id, target_layer)
            
            # 更新当前层节点为下一层的候选
            current_layer_nodes = selected_nodes
            stats["layers_built"] += 1
        
        # 第四步：更新最大层数和entry_point
        # 找到实际构建的最高层
        actual_max_level = max(stats["nodes_by_layer"].keys())
        self.max_level = actual_max_level
        
        # 随机选择最高层的一个节点作为entry_point
        highest_layer_nodes = []
        for node_id, node in self.items.items():
            if node.level == actual_max_level:
                highest_layer_nodes.append(node_id)
        
        if highest_layer_nodes:
            self.entry_point = random.choice(highest_layer_nodes)
            stats["entry_point"] = self.entry_point
            stats["entry_point_level"] = actual_max_level
        
        print(f"重建完成，最大层数: {actual_max_level}, 入口点: {self.entry_point}")
        print(f"重建的层: {[layer for layer in range(1, actual_max_level + 1)]}")
        
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
