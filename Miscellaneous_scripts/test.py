import pandas as pd
import multiprocessing as mp
import datetime
from netwulf import visualize
import pickle
import networkx as nx
import json
import os
import sys

import NYC_tools.utils_nyc

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import Topology.utils_topology as utils
import NYC_tools.NYC_data_prep_functions as nyc_tools
from ExMAS.main_prob_coeffs_OLD import main as exmas_algo
from ExMAS.utils import make_graph as exmas_make_graph

topological_config = utils.get_parameters(
    r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\configs\topology_settings3.json"
)
params = nyc_tools.get_config(
    r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\configs\nyc_prob_coeffs.json")
params = utils.update_probabilistic(topological_config, params)

# with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\06-10-22\all_graphs_list_06-10-22.obj",
#           'rb') as file:
#     dotmap_list_results = pickle.load(file)

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\06-10-22\all_graphs_list_06-10-22.obj",
          'rb') as file:
    all_graphs_list = pickle.load(file)

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\06-10-22\dotmap_list_06-10-22.obj",
          'rb') as file:
    dotmaps_list_results = pickle.load(file)

topological_config.path_results = r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\06-10-22"
utils.analysis_all_graphs(all_graphs_list, topological_config)
topo_params = topological_config

# pool = mp.Pool(mp.cpu_count())
# graph_list = [pool.apply(exmas_make_graph, args=(data.sblts.requests, data.sblts.rides)) for data in
#               dotmaps_list_results]
# topological_stats = [utils.GraphStatistics(graph, "INFO") for graph in graph_list]
# topo_dataframes = pool.map(utils.worker_topological_properties, topological_stats)
# pool.close()

graph_list = [exmas_make_graph(data.sblts.requests, data.sblts.rides) for data in dotmaps_list_results]
topological_stats = [utils.GraphStatistics(graph, "INFO") for graph in graph_list]
topo_dataframes = [utils.worker_topological_properties(x) for x in topological_stats]
i = 1
settings = []
for j in range(1000):
    for k in range(len(topo_params['values'])):
        params[topo_params['variable']] = topo_params['values'][k]
        settings.append({'Replication': j, 'Batch': i, topo_params.variable: topo_params['values'][k]})

settings_list = settings

""" Merge results """
merged_results = utils.merge_results(dotmaps_list_results, topo_dataframes, settings_list)
merged_file_path = topological_config.path_results + 'merged_files_' + \
                   str(datetime.date.today().strftime("%d-%m-%y")) + '.xlsx'
merged_results.to_excel(merged_file_path, index=False)

""" Compute final results """
variables = ['Batch']
utils.APosterioriAnalysis(pd.read_excel(merged_file_path),
                          topological_config.path_results,
                          topological_config.path_results + "temp/",
                          variables,
                          topological_config.graph_topological_properties,
                          topological_config.kpis,
                          topological_config.graph_properties_against_inputs,
                          topological_config.dictionary_variables).do_all()



