from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import numpy as np

start = time.process_time()

for j in np.arange(0, 1, 0.00001):
    x = np.exp(j)/(np.exp(j)+np.exp(1))

print(time.process_time() - start)

start = time.process_time()

for j in np.arange(0, 1, 0.00001):
    x = norm.cdf(j)

print(time.process_time() - start)