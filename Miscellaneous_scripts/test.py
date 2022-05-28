import numpy as np

from ExMAS.main_prob import main as exmas_algo
from NYC_tools import NYC_data_prep_functions as nyc_func
import Topology.utils_topology as utils
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import datetime

num_list = [1] + list(range(100, 1000, 100))

topological_config = utils.get_parameters('data/configs/topology_settings.json')
utils.create_results_directory(topological_config)

fig, axes = plt.subplots(nrows=2, ncols=5, sharex='col', sharey='row')

for col in range(2):
    for row in range(5):
        num = num_list[5*col+row]
        axes[col, row].imshow(topological_config.path_results + "temp/graph_" + str(datetime.date.today().strftime("%d-%m-%y"))
                    + "_no_" + num + ".png")

plt.show()

