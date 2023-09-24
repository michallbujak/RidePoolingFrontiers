import pandas as pd
import osmnx as ox
import networkx as nx
from dotmap import DotMap


# short_df = pd.read_csv(r"C:\Users\szmat\Downloads\OpenDataDC_Taxi_2022\taxi_202201.csv")
params = DotMap()
params.city = "Washington, D.C., USA"
params.dist_threshold = 10000
params.paths.G = r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\ExMAS\data\graphs\WashingtonDC.graphml"
params.paths.skim = r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\ExMAS\data\graphs\WashingtonDC.csv"

dm = DotMap()

def download_G(inData, _params, make_skims=True):
    # uses osmnx to download the graph
    inData.G = ox.graph_from_place(_params.city, network_type='drive')
    inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
    if make_skims:
        inData.skim_generator = nx.all_pairs_dijkstra_path_length(inData.G,
                                                                  weight='length')
        inData.skim_dict = dict(inData.skim_generator)  # filled dict is more usable
        inData.skim = pd.DataFrame(inData.skim_dict).fillna(_params.dist_threshold).T.astype(
            int)  # and dataframe is more intuitive
    return inData

def save_G(inData, _params, path=None):
    # saves graph and skims to files
    ox.save_graphml(inData.G, filepath=_params.paths.G)
    inData.skim.to_csv(_params.paths.skim, chunksize=20000000)

dm = download_G(dm, params)
save_G(dm, params)


