import numpy as np
from numba import jit

@jit
def test_foo():
    x = np.array([0.02, 0.01])
    return [1]*len(x)

print(test_foo())
