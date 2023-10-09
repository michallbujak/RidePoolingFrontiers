import pickle
import matplotlib.pyplot as plt
import pandas as pd

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\results_chicago.pickle", "rb") as file:
    x = pickle.load(file)

plt.hist([len(t['requests']) for t in x])
plt.show()
plt.close()
