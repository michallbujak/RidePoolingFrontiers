import pickle

with open(r"C:\Users\zmich\Documents\GitHub\ExMAS_sideline\Miscellaneous_scripts\data\29-11-22\single_full_dotmap_29-11-22.obj", "rb") as file:
       results=pickle.load(file)

print(results[0].G.nodes._nodes)
