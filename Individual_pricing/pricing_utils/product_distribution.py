import secrets
from bisect import bisect
import itertools
import random
import numpy as np


def sample_multinormal(
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

    count = 0
    while count < size:
        sampled_uniform = rng.random()
        index = bisect(probs, sampled_uniform)

        out.append(rng.normal(loc=means[index], scale=st_devs[index]))
        count += 1

    return out


class ProductDistribution:
    def __init__(self):
        self.samples = []
        self.sample = None
        self.size = None

    def new_sample(self, distribution_type: str, **kwargs):
        assert distribution_type in ["multinormal"], \
            "not accepted distribution type"

        assert distribution_type == "multinormal" and \
               all([t in kwargs.keys() for t in
                    ["probs", "means", "st_devs", "size", "seed"]]), \
                "Make sure all of the ['probs', 'means'," \
                " 'st_devs', 'size', 'seed'] are passed"

        self.samples.append(sample_multinormal(
            probs=kwargs.get("probs"),
            means=kwargs.get("means"),
            st_devs=kwargs.get("st_devs"),
            size=kwargs.get("size"),
            seed=kwargs.get("seed")
        ))

        if self.size is None:
            self.size = kwargs.get("size")
        else:
            self.size *= kwargs.get("size")

        self.cumulative_sample()

    def remove_sample(self, index: int):
        del self.samples[index]
        self.cumulative_sample()

    def cumulative_sample(self):
        if len(self.samples) == 0:
            self.sample = None
            self.size = 0

        else:
            sampled = itertools.product(*self.samples)
            sampled = sorted([np.prod(t) for t in sampled])
            self.sample = sampled
            self.size = len(self.sample)

    def cdf(self, argument):
        assert isinstance(self.size, int), "Incorrect size"

        return bisect(self.sample, argument) / self.size

    def sample_value(self):
        return random.sample(self.sample)

    def inverse_cdf(self, argument):
        return self.sample[int(argument * self.size)]


# Betas = ProductDistribution()
# Betas.new_sample(
#     distribution_type="multinormal",
#     probs=[0.2, 0.1, 0.7],
#     means=[1, 2, 3],
#     st_devs=[0.2, 0.3, 0.4],
#     size=1000,
#     seed=123
# )
#
# Betas.new_sample(
#     distribution_type="multinormal",
#     probs=[0.4, 0.6],
#     means=[1, 2],
#     st_devs=[0.1, 0.1],
#     size=1000,
#     seed=124
# )
#
# Betas.cumulative_sample()
# z = 0
# import matplotlib.pyplot as plt
# plt.hist(Betas.sample)
# plt.show()
