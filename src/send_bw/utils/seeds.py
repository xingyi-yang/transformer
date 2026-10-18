"""Deterministic, name-addressed seed streams.

Every consumer asks for a generator by a stable name, so adding a new consumer
never shifts the random numbers seen by existing ones.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_hash(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


class SeedStream:
    def __init__(self, root_seed: int):
        self.root_seed = int(root_seed)

    def seed(self, *names: object) -> int:
        key = tuple(stable_hash(str(n)) for n in names)
        ss = np.random.SeedSequence(self.root_seed, spawn_key=key)
        return int(ss.generate_state(1, dtype=np.uint32)[0])

    def rng(self, *names: object) -> np.random.Generator:
        key = tuple(stable_hash(str(n)) for n in names)
        return np.random.default_rng(np.random.SeedSequence(self.root_seed, spawn_key=key))
