import visualising_functions as vis_eff

date = "25-10-22"
sblts_exmas = "exmas"
special_name = "_maxi"

config = vis_eff.config_initialisation('data/configs/topology_settings3.json', date)
config.path_results = 'data/results/' + date + special_name + '/'
config.date = date

vis_eff.create_figs_folder(config)

rep_graphs, dotmap_list, all_graphs_list = vis_eff.load_data(config)

vis_eff.graph_visualisation_with_netwulf(all_graphs_list, rep_graphs)

vis_eff.kpis_gain(dotmap_list, config)

vis_eff.probability_of_sharing_classes(dotmap_list, config)

