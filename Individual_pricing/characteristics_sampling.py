import itertools as it
import numpy as np
from bisect import bisect


def sample_normal(
        probs: list or tuple,
        means: list or tuple,
        st_devs: list or tuple,
        size: int,
        seed: int
) -> list:
    out = []
    rng = np.random.default_rng(seed)
    if probs[-1] != 1:
        probs = np.cumsum(probs)

    assert probs[-1] == 1, "Incorrect probs parameter"

    for
