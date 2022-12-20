import pickle

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\19-12-22\dotmap_list_19-12-22.obj", "rb") as file:
       results=pickle.load(file)

print(results[0].G.nodes._nodes)
