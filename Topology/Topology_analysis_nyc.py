import utils_topology as utils
import NYC_tools.NYC_data_prep_functions as nyc_tools
from ExMAS.main_prob import main as exmas_algo


if __name__ == "__main__":
    """ Load all the topological parameters """
    topological_config = utils.get_parameters('data/configs/topology_settings.json')

    """ Set up varying parameters (optional) """
    topological_config.variable = ['shared_discount']
    topological_config.values = [0.22, 0.24]

    """ Run parameters """
    topological_config.replications = 2
    topological_config.no_batches = 3

    """ Prepare data """
    dotmaps_list, params = nyc_tools.prepare_batches(topological_config.no_batches,
                                                     filter_function=lambda x: len(x.requests) > 20,
                                                     config_name="nyc_prob")

    res = nyc_tools.run_exmas_nyc_batches(exmas_algo, params, dotmaps_list, topological_config,
                                          replications=topological_config.replications)

