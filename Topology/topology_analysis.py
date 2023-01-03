import pickle
import numpy as np
import pandas as pd
import Utils.utils_topology as utils_topology
import multiprocessing as mp
import Utils.visualising_functions as vf

if __name__ == "__main__":
    """ Load all the topological parameters """
    topological_config = utils_topology.get_parameters('data/configs/topology_settings_like_old.json')

    """ Set up varying parameters (optional) """
    topological_config.variable = 'shared_discount'
    topological_config.values = np.round(np.arange(0.1, 0.12, 0.01), 2)

    utils_topology.create_results_directory(topological_config, date="20-12-22")

    with open(
            r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\20-12-22\all_graphs_list_20-12-22.obj",
            "rb") as file:
        graph_list = pickle.load(file)

    with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\20-12-22\dotmap_list_20-12-22.obj",
              "rb") as file:
        data = pickle.load(file)

    graph_list = [g['bipartite_shareability'] for g in graph_list]
    graph_list_obj = [utils_topology.GraphStatistics(graph, "CRITICAL") for graph in graph_list]

    pool = mp.Pool(mp.cpu_count())
    graph_list_obj_calculated = pool.map(utils_topology.worker_topological_properties, graph_list_obj)
    pool.close()

    row_names = list(graph_list_obj_calculated[0].index) + ["shared_discount"] + list(data[0]["exmas"]["res"].index)[:-2]
    df = pd.DataFrame()

    for d, s_d, res in zip(graph_list_obj_calculated, topological_config["values"], data):
        d_to_append = pd.concat([d, pd.DataFrame({d.columns[0]: [s_d]})], ignore_index=True)
        d_to_append = d_to_append.append(pd.DataFrame(res["exmas"]["res"])[:-2], ignore_index=True)
        d_to_append.index = row_names
        df = pd.concat([df, d_to_append.T])

    df.reset_index(inplace=True, drop=True)
    x = 0

