import utils_topology as utils
import NYC_tools.NYC_data_prep_functions as nyc_tools
from ExMAS.main_prob import main as exmas_algo
from ExMAS.utils import make_graph as exmas_make_graph

import pandas as pd
import multiprocessing as mp
import os

if __name__ == "__main__":
    """ Load all the topological parameters """
    print(os.getcwd())
    topological_config = utils.get_parameters('data/configs/topology_settings.json')

    """ Set up varying parameters (optional) """
    topological_config.variable = 'shared_discount'
    topological_config.values = [0.22, 0.24]

    """ Run parameters """
    topological_config.replications = 2
    topological_config.no_batches = 2

    """ Prepare data """
    dotmaps_list, params = nyc_tools.prepare_batches(topological_config.no_batches,
                                                     filter_function=lambda x: len(x.requests) > 20,
                                                     config=topological_config.initial_parameters)

    dotmaps_list_results, settings_list = nyc_tools.run_exmas_nyc_batches(exmas_algo, params, dotmaps_list,
                                                                          topological_config,
                                                                          replications=topological_config.replications)

    """ Perform topological analysis """
    pool = mp.Pool(mp.cpu_count())
    graph_list = [pool.apply(exmas_make_graph, args=(data.sblts.requests, data.sblts.rides)) for data in
                  dotmaps_list_results]
    topological_stats = [utils.GraphStatistics(graph, "INFO") for graph in graph_list]
    topo_dataframes = pool.map(utils.worker_topological_properties, topological_stats)
    pool.close()

    """ Merge results """
    x = 0
    def merge_results(dotmaps_list_results, topo_dataframes, settings):
        x = 0
