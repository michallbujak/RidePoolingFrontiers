import pickle
import networkx as nx
import netwulf as nw
import matplotlib.pyplot as plt

with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\10-01-23\dotmap_list_10-01-23.obj", "rb") as file:
       e = pickle.load(file)

res = e[0]["exmas"].res
x = 0
