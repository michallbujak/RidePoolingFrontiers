import numpy as np
import secrets
from bisect import bisect
import itertools
import functools


def sample_normal(
        probs: list or tuple,
        means: list or tuple,
        st_devs: list or tuple,
        size: int,
        seed: int
) -> list:
    out = []
    rng = np.random.default_rng(secrets.randbits(seed))
    if probs[-1] != 1:
        probs = np.cumsum(probs)

    assert probs[-1] == 1, "Incorrect probs parameter"

    for j in range(size):
        z = rng.random()
        index = bisect(probs, z)

        out.append(rng.normal(loc=means[index], scale=st_devs[index]))

    return out


def product_distribution(
        probs: (list or tuple, list or tuple),
        means: (list or tuple, list or tuple),
        st_devs: (list or tuple, list or tuple),
        individual_sizes: int,
        seed: (int, int) = (123, 124)
):
    sample_a = sample_normal(
        probs=probs[0],
        means=means[0],
        st_devs=st_devs[0],
        size=individual_sizes,
        seed=seed[0]
    )

    sample_b = sample_normal(
        probs=probs[1],
        means=means[1],
        st_devs=st_devs[1],
        size=individual_sizes,
        seed=seed[1]
    )

    sample = [a*b for a, b in itertools.product(sample_a, sample_b)]
    sample = sorted(sample)

    def modified_bisect(a, t, s):
        return bisect(a, t)/s

    return functools.partial(modified_bisect, a=sample, s=individual_sizes**2)


func = product_distribution(
    ([0.2, 0.1, 0.7], [0.4, 0.6]),
    ([1, 2, 3], [1, 2]),
    ([0.2, 0.3, 0.4], [0.1, 0.1]),
    100
)

z = 0
