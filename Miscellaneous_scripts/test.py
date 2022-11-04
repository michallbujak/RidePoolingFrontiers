import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = {
       "A": pd.DataFrame({"X": [0, 0, 0, 2, 4]}),
       "B": pd.DataFrame({"X": [0, 0, 0, 0, 1, 2]}),
       "C": pd.DataFrame({"X": [0, 0, 2, 4]})
}

datasets = [t["X"] for t in [v for k, v in data.items()]]
labels = [k for k, v in data.items()]

bins = [0, 2, 4, np.inf]

plt.hist(datasets, density=True, histtype='step', alpha=0.6, bins=bins, label=labels, cumulative=True)
plt.legend(loc="upper right")
plt.show()
