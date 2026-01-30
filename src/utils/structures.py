import numpy as np
from dataclasses import dataclass

@dataclass
class SuperPoint:
    id: int
    centroid: np.ndarray
    pca_dir: np.ndarray
    thickness: float
    verticality: float
    n_points: int
    bbox_radius: float
    chunk_id: int


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        else:
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1