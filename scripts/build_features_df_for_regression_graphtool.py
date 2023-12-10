import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graph_tool.all import *
from functools import reduce
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datetime import datetime as dt
from datetime import timedelta

import frame_stats as fs
import frame_stats.time_series as ts
import frame_stats.causal_inferrence as ci  # has the functions for setting up regression


# config has frame names
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# paths to data files ( need to fix this)
with open("workflow/sample_paths.json", "r") as pf:
    paths = json.loads(pf.read())

# load all of the frames and tweet time stamps etc.
f = pd.read_csv(paths["all_frames"], sep="\t")
filtered_tweets = fs.filter_users_by_activity(f, 10)

# FOR TESTING
filtered_tweets = filtered_tweets.sample(frac=0.1)

# connect screen names and user ids. needed to work with mention network
user_id_map = pd.read_csv(paths["public"]["user_id_map"], sep="\t",
                          dtype={"screen_name": str, "user_id": str})

print("building name to id mappers")
name_to_id = {row["screen_name"]: row["user_id"] for _, row in user_id_map.iterrows()}
id_to_name = {row["user_id"]: row["screen_name"] for _, row in user_id_map.iterrows()}
print("name to id mappers built")

# the mention network itself
mentions = pd.read_csv(paths["mentions"]["network"], sep="\t",
                       dtype={"uid1": str, "uid2": str,
                              "1to2freq": int, "2to1freq": int})


# list of all frame names
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]

# public tweet metadata like ideology scores, engagement, etc.
meta = (pd.read_csv(paths["public"]["metadata"], sep="\t")
          .drop(all_frame_list + ["Unnamed: 0", "Hero", "Threat", "Victim"], 
                axis="columns"))
meta_subset = pd.merge(filtered_tweets[["id_str"]], meta, how="left", on="id_str")
meta_subset["id_str"] = meta_subset["id_str"].astype(str)

#do the real shit
print("building graph")
g = Graph()
g.add_edge_list(mentions[["uid1", "uid2"]].values, hashed=True)
print("graph built")

# building adjacency list
adj_list = {}
print("building adjacency list")
for vert in tqdm(g.vertices()):
    print("\n")
    print(vert.id)
    print("\n")
    print(vert)
    print("\n")
    print(type(vert))
    exit(1)
    if vert in id_to_name:
        adj_list[id_to_name[vert]] = []
        for neighbor in vert.out_neighbors():
            if neighbor in id_to_name:
                adj_list.append(id_to_name[neighbor])
print("adjacency list done")

def get_unique_mentions(user: str,
                        mentions: graph_tool.Graph,
                        name_to_id_map: Dict,
                        id_to_name_map: Dict) -> Optional[Tuple[int, List[str]]]:
    # if we have the user in our map we continue doing stuff
    if user in name_to_id_map:
        user_id = name_to_id_map[user]
        #user_mentions = mentions.get_all_neighbors(user_id)
        user_mentions = np.array([1, 2, 3])
        unique_mentions = user_mentions.shape[0]
        
        neighbors = []
        for id in user_mentions:

            if id in id_to_name_map:

            # again we might not have the user in our map
                neighbors.append(id_to_name_map[id])

        return (unique_mentions, neighbors)

    else:
         return None

print("searching neighbors")
feature_rows = []
mention_neighbors = {}
for user in tqdm(filtered_tweets["screen_name"].unique()):
    row = {}
    row["screen_name"] = user

    lookup = get_unique_mentions(user, g, name_to_id, id_to_name)

    if lookup:
        count, neighbors = lookup
        row["mention_degree"] = count

        mention_neighbors["screen_name"] = neighbors

    else:
        row["mention_degree"] = 0
        mention_neighbors["screen_name"] = []


features_df = pd.merge(meta_subset, pd.DataFrame(feature_rows), how="left", on="screen_name")
features_df.to_csv(paths["regression"]["features"], sep="\t", index=False)

with open(paths["mentions"]["neighbors"], "w") as neighbors_out:
    json.dump(mention_neighbors, neighbors_out)

# %%
