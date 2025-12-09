import pandas as pd
import multiprocessing as mp
import datetime
from netwulf import visualize
import pickle
import networkx as nx
import json
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import Utils.utils_topology as utils
import NYC_tools.NYC_data_prep_functions as nyc_tools
from ExMAS.probabilistic_exmas import main as exmas_algo
from ExMAS.utils import make_graph as exmas_make_graph

os.chdir(os.path.dirname(os.getcwd()))

if __name__ == "__main__":
    """ Load all the topological parameters """
    topological_config = utils.get_parameters('Topology/data/configs/topology_settings_rev.json')

    """ Set up varying parameters (optional) """
    # topological_config.variable = 'shared_discount'
    # topological_config.values = [0.22, 0.24]

    """ Run parameters """
    topological_config.replications = 20
    topological_config.no_batches = 1

    """ Prepare folder """
    topological_config.path_results = "Topology/" + topological_config.path_results
    utils.create_results_directory(topological_config)

    output = {}
    """ Prepare data """
    for batch_size in range(100, 201, 1):
        dotmaps_list, params = nyc_tools.prepare_batches(topological_config.no_batches,
                                                         filter_function=lambda x: len(x.requests) == batch_size,
                                                         config=topological_config.initial_parameters)

        # for prob in np.arange(0, 0.42, 0.03):
        #     dotmaps_list, exmas_params = nyc_tools.prepare_batches(topological_config.no_batches,
        #                                                      filter_function=lambda x: len(x.requests) == 147,
        #                                                      config=topological_config.initial_parameters)

        """ Run ExMAS """
        params = utils.update_probabilistic(topological_config, params)
        params.multinormal_probs = (0.29, 0.57, 0.81, 1)
        # exmas_params.multinormal_probs = (0.37-(prob/3), 0.37+0.36-(2*prob/3), 0.37+0.36+(prob/3), 1)
        params.multinormal_args = (
            ((16.98 / 3600, 1.22), (0.31765 / 3600, 0.0815)),
            ((14.02 / 3600, 1.135), (0.2058 / 3600, 0.07056)),
            ((26.25 / 3600, 1.049), (5.7765 / 3600, 0.06027)),
            ((7.78 / 3600, 1.18), (1 / 3600, 0.07626))
        )
        params.type_of_distribution = "multinormal"
        params.sampling_function_with_index = True

        dotmaps_list_results, settings_list = nyc_tools.testing_exmas_basic(exmas_algo, params, dotmaps_list,
                                                                            topo_params=topological_config,
                                                                            replications=topological_config.replications,
                                                                            logger_level='INFO',
                                                                            sampling_function_with_index=True)

        output[batch_size] = [t["exmas"]["res"]["VehHourTrav_ns"] -
                              t["exmas"]["res"]["VehHourTrav"]
                              for t in dotmaps_list_results]

        # output[prob] = [dotmaps_list_results]

    with open("critical_mass_data2.pickle", 'wb') as file:
        pickle.dump(output, file)

    # with open("sensitivity_classes.pickle", 'wb') as file:
    #     pickle.dump(output, file)
