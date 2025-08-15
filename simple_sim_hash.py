import numpy as np
from typing import List


class SimpleSimHash:
    def __init__(self, dim: int, n_bits: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((dim, n_bits)).astype(np.float32)
        self.n_bits = n_bits
        self.buckets = {}  # dict[int -> list[int]]

    def hash(self, v: np.ndarray) -> int:
        signs = (v @ self.planes) > 0
        # pack to int
        h = 0
        for i, s in enumerate(signs):
            if s: h |= (1 << i)
        return h

    def add(self, vid: int, vec: np.ndarray):
        h = self.hash(vec)
        lst = self.buckets.setdefault(h, [])
        if len(lst) < 20000:   # 每桶上限，防止爆炸（可调/采样）
            lst.append(vid)

    def get_near(self, vec: np.ndarray, radius: int = 1) -> List[int]:
        base = self.hash(vec)
        ids = set(self.buckets.get(base, []))
        if radius >= 1:
            for i in range(self.n_bits):
                ids |= set(self.buckets.get(base ^ (1 << i), []))
        # 半径可以继续扩展
        return list(ids)