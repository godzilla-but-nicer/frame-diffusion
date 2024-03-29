import json
from tqdm import tqdm
import pandas as pd

import frame_stats as fs
import frame_stats.time_series as ts
import frame_stats.causal_inferrence as ci  # has the functions for setting up regression
import pickle

print("loading config and paths")
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# paths to data files ( need to fix this)
with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())
print("config and paths loaded")


print("loading id map")
user_id_map = pd.read_csv(paths["public"]["user_id_map"], sep="\t",
                          dtype={"screen_name": str, "user_id": str})
print("id map loaded")

print("building screen name lookup")
name_lookup = {}
for _, row in tqdm(user_id_map.iterrows()):
    name_lookup[row["user_id"]] = row["screen_name"]
print("screen name lookup built")

print("loading adjacency list")
with open(paths["mentions"]["adjacency_list_ids"], "r") as fout:
    mention_neighbors = json.loads(fout.read())
print("adjacency list loaded")


print("converting user ids to screen names")
new_mention_neighbors = {}
for user_id in mention_neighbors.keys():

    if user_id in name_lookup:
        screen_name = name_lookup[user_id]
        new_mention_neighbors[screen_name] = []

        for neighbor in mention_neighbors[user_id]:
            if neighbor in name_lookup:
                neighbor_screen_name = name_lookup[neighbor]
                new_mention_neighbors[screen_name].append(neighbor_screen_name)

mention_neighbors = new_mention_neighbors
print("conversion complete")

with open(paths["mentions"]["adjacency_list"], "w") as fnames:
    json.dump(mention_neighbors, fnames)
