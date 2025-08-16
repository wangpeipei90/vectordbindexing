# from __future__ import annotations

import math
import random
import queue
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
                self.neighbours[layer][source] = [n for n, _ in distances[:self.M]]

    def _add_QD_link(self, source: int, dest: int, layer: int, reset_: bool) -> None:
        """Add a directed link from ``source`` to ``dest`` on ``layer`` 不截断"""
        nbrs = self.neighbours[layer][source]
        if dest not in nbrs:
            nbrs.append(dest)
        if reset_ and len(nbrs) > self.M:
            # 计算到 source 的距离并保留最近的 M 个
            distances = [(n, self.distance(self.items[source].vector, self.items[n].vector))
                            for n in nbrs]
            distances.sort(key=lambda x: x[1])
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
        """Return (closest_id, steps); steps is a list of dicts describing moves."""
        steps = []
        current = entry_id
        best_dist = self.distance(vec, self.items[current].vector)
        steps.append(current)
        improved = True
        while improved:
            improved = False
            for nb in self.neighbours[layer][current]:
                dist = self.distance(vec, self.items[nb].vector)
                steps.append(nb)
                if dist < best_dist:
                    best_dist = dist
                    current = nb
                    improved = True
        return current, steps

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

    def query(self, vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Return the k nearest neighbours to vector (id, distance)."""
        if not self.items:
            return []
        vec = _unit_norm(np.asarray(vector, dtype=np.float32))
        curr = self.entry_point
        for l in range(self.max_level, 0, -1):
            curr = self._search_layer_greedy(vec, curr, l)
        candidates = self._search_layer(vec, curr, 0, max(self.ef_search, k))
        dists = [(cid, self.distance(vec, self.items[cid].vector)) for cid in candidates]
        dists.sort(key=lambda x: x[1])
        return dists[:k]

    # ===== Public API: search with steps to reach a target =====
    def search_steps_to_target(self, vector: np.ndarray, target_id: int, k: int = 10, ef: Optional[int] = None):
        """Run a query and return the detailed navigation steps until the target is first reached.
        Returns:
            {
              "found": bool,
              "trace": [ ... step dicts ... ],
              "final_candidates": List[Tuple[id, dist]]  # top-k at the end (optional)
            }
        """
        if not self.items:
            return {"found": False, "trace": [], "final_candidates": []}

        vec = _unit_norm(np.asarray(vector, dtype=np.float32))
        trace = []

        # 1) run best-first with trace until target first appears/popped
        curr = self.entry_point
        eff = max(self.ef_search, k) if ef is None else max(ef, k)
        steps, found = self._search_layer_trace_until_target(vec, curr, 0, eff, target_id)
        trace.extend(steps)

        # 3) Optionally finish a normal search to return a final top-k list
        # （这部分不影响“找到 target 的时刻”，仅用于返回参考的 top-k）
        cands = self._search_layer(vec, curr, 0, eff)
        dists = [(cid, self.distance(vec, self.items[cid].vector)) for cid in cands]
        dists.sort(key=lambda x: x[1])
        topk = dists[:k]

        return {"found": found, "trace": trace, "final_candidates": topk}

    # ====== Edge augmentation from offline query→topk ======
    def augment_from_query_topk(
        self,
        query_topk: Dict[int, list[int]],
        strategy: str = "star",
        layer: int = 0,
        max_new_edges_per_node: int = 4,
        dedup: bool = True,
        strict_exist_check: bool = True,
        reset_: bool = False,
    ) -> dict:
        """
        用离线 query→topK 结果增强图（支持三种策略：star/clique/projection）。
        - projection: 邻域感知投影（RoarGraph 风格的轻量实现）
        * pivot = top1
        * 候选按 rank 顺序（如有 self._dist 可做更稳排序）
        * occlusion 规则：仅保留对 pivot 更“独立”的候选，避免互相遮挡
        可调属性（若未设置则使用默认）:
        - self.qd_occlude_alpha: float, 默认 1.0（=严格版 occlusion）
        - self.qd_use_metric: bool, 默认 True 且要求存在 self._dist(u,v)
        - self.qd_chain: int, 默认 0；>0 时对已选邻居做少量“链式”互连以增强可达
        """
        assert strategy in ("star", "clique", "projection")
        if layer < 0 or layer > self.max_level:
            return {"pairs_considered": 0, "pairs_added": 0, "skipped_missing": 0, "skipped_existing": 0, "pruned_by_cap": 0, "skipped_occluded": 0}

        # 读取可选开关（不破坏签名）
        occlude_alpha: float = float(getattr(self, "qd_occlude_alpha", 1.0))   # >=1.0
        use_metric: bool = bool(getattr(self, "qd_use_metric", True))
        chain_extra: int = int(getattr(self, "qd_chain", 0))  # 每个查询对已选邻居额外链几条边（小数值即可）

        # 每个节点本次新增计数（限流）
        from collections import defaultdict
        added_per_node: Dict[int, int] = defaultdict(int)

        stats = {
            "pairs_considered": 0,
            "pairs_added": 0,
            "skipped_missing": 0,
            "skipped_existing": 0,
            "pruned_by_cap": 0,
            "skipped_occluded": 0,
        }

        def has_capacity(u: int) -> bool:
            return (max_new_edges_per_node is None) or (added_per_node[u] < max_new_edges_per_node)

        def can_add(u: int, v: int):
            # 度预算
            if (max_new_edges_per_node is not None) and ( (added_per_node[u] >= max_new_edges_per_node) or (added_per_node[v] >= max_new_edges_per_node) ):
                stats["pruned_by_cap"] += 1
                return False, 0
            for layer_id in range(0, self.max_level):
                if u not in self.neighbours[layer_id]:
                    continue
                # 去重
                if v in self.neighbours[layer_id][u]:
                    stats["skipped_existing"] += 1
                    return False, layer_id
                # 度限制：通过u能够k跳内去到v
                can_add = True
                degree_queue = queue.Queue()
                sub_degree_queue = queue.Queue()
                explore_depth = 3
                degree_queue.put({"layer":layer_id, "id": u})
                while degree_queue.empty() and sub_degree_queue.empty():
                    if explore_depth % 2 == 1:
                        m = degree_queue.get()
                        for x in self.neighbours[m["layer"]][m["id"]]:
                            for ano_layer in range(0, self.max_level):
                                if x not in self.neighbours[ano_layer]:
                                    continue
                                if v in self.neighbours[ano_layer][x]:
                                    can_add = False
                                    break
                                sub_degree_queue.put({"layer":ano_layer, "id": x})
                    else:
                        m = sub_degree_queue.get()
                        for x in self.neighbours[m["layer"]][m["id"]]:
                            for ano_layer in range(0, self.max_level):
                                if x not in self.neighbours[ano_layer]:
                                    continue
                                if v in self.neighbours[ano_layer][x]:
                                    can_add = False
                                    break
                                if explore_depth > 0:
                                    degree_queue.put({"layer":ano_layer, "id": x})
                    if not can_add:
                        break
                    if degree_queue.empty() or sub_degree_queue.empty():
                        explore_depth -= 1

                if can_add:
                    return True, layer_id
                else:
                    return False, layer_id

            return False, layer

        def _dist(u: int, v: int) -> float:
            # 若类里提供了度量函数则用之，否则退化为“同 rank 下不做度量比较”（返回+∞使得不过滤）
            if use_metric and hasattr(self, "_dist"):
                return float(self._dist(u, v))
            return float("inf")

        def _add_pair(u: int, v: int, reset_, layer_id):
            # 双向加
            self._add_QD_link(u, v, layer_id, reset_)
            self._add_QD_link(v, u, layer_id, reset_)
            added_per_node[u] += 1
            added_per_node[v] += 1
            stats["pairs_added"] += 1

        # --- projection 策略的核心：AcquireNeighbors（RoarGraph Alg.3 的轻量实现）---
        def acquire_neighbors(pivot: int, candidates: list[int], cap: int) -> list[int]:
            """
            从 candidates 里选一组“对 pivot 有帮助且相互不过度遮挡”的点，最多 cap 个。
            规则：按 candidates 顺序（≈按与 query 的距离从小到大），
                仅当 对所有已选 r:  dist(c, r) > occlude_alpha * dist(c, pivot) 时保留 c。
            """
            picked: list[int] = []
            for c in candidates:
                if c == pivot:
                    continue
                if cap is not None and len(picked) >= cap:
                    break
                if not has_capacity(pivot) or not has_capacity(c):
                    continue
                ok = True
                if use_metric and picked:
                    dcp = _dist(c, pivot)
                    # 若没有度量（inf），不会触发剔除；有度量时执行遮挡判断
                    if dcp != float("inf"):
                        for r in picked:
                            if _dist(c, r) <= occlude_alpha * dcp:
                                ok = False
                                break
                if ok:
                    picked.append(c)
            return picked

        for _qid, topk in query_topk.items():
            # 清洗：只保留在索引中的 id
            ids = [nid for nid in topk if nid in self.items]
            if strict_exist_check and len(ids) < len(topk):
                stats["skipped_missing"] += (len(topk) - len(ids))
            if len(ids) <= 1:
                continue

            if strategy == "star":
                hub = ids[0]
                for v in ids[1:]:
                    stats["pairs_considered"] += 1
                    can_add_, layer_id = can_add(hub, v)
                    if can_add_:
                        _add_pair(hub, v, reset_, layer_id)

            elif strategy == "clique":
                n = len(ids)
                for i in range(n):
                    for j in range(i + 1, n):
                        u, v = ids[i], ids[j]
                        stats["pairs_considered"] += 1
                        can_add_, layer_id = can_add(u, v)
                        if can_add_:
                            _add_pair(u, v, reset_, layer_id)

            elif strategy == "projection":
                # 1) pivot 选 top1（RoarGraph 中为“最邻近的 base 节点”）
                pivot = ids[0]
                cand = ids[1:]  # 如有 query 距离，可在外部保证 topk 已按距离升序

                # 2) 邻域感知筛选（多样性/遮挡规则）
                picked = acquire_neighbors(pivot, cand, max_new_edges_per_node)

                # 3) 连接：pivot ↔ picked
                for v in picked:
                    stats["pairs_considered"] += 1
                    can_add_, layer_id = can_add(pivot, v)
                    if can_add_:
                        _add_pair(pivot, v, reset_, layer_id)
                    else:
                        stats["skipped_occluded"] += 0  # 占位，保持字段

                # 4) 可选的小幅“链式”增强：在 picked 里顺次连少量边，提升可达性（RoarGraph 的 connectivity enhancement 的轻量影子）
                if chain_extra > 0 and len(picked) > 1:
                    # 只连前若干条顺次对
                    limit = min(chain_extra, len(picked) - 1)
                    for i in range(limit):
                        u, v = picked[i], picked[i + 1]
                        can_add_, layer_id = can_add(u, v)
                        if can_add_:
                            _add_pair(u, v, reset_, layer_id)
                            stats["pairs_considered"] += 1
        return stats
