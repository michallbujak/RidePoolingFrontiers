import pandas as pd
import numpy as np
import pickle
import scipy.stats.norm as norm

# df = pd.DataFrame({"A": [1, 2], "b": [3, 4], "c": [5, 6]})


def inverse_normal(means, stds):
    def internal_function(x):
        return [norm.ppf(x, mean, std) for mean, std in zip(means, stds)]
    return internal_function

pr = inverse_normal([1], [0])


print(pr(0))
