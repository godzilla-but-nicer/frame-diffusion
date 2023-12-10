import json
import numpy as np
import pandas as pd
import statsmodels.api as sm  # does the regression
import matplotlib.pyplot as plt
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
with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())

# load all of the frames and tweet time stamps etc.
f = pd.read_csv(paths["all_frames"], sep="\t", dtype={"id_str": str})
filtered_tweets = fs.filter_users_by_activity(f, 10)

# FOR TESTING
filtered_tweets = filtered_tweets.sample(frac=1)

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

# make long
print("making mentions longer")
top_half = mentions[["uid1", "uid2"]].rename({"uid1": "source",
                                                  "uid2": "target"},
                                                  axis="columns")
bottom_half = mentions[["uid2", "uid1"]].rename({"uid2": "source",
                                                     "uid1": "target"},
                                                  axis="columns")
mentions = pd.concat((top_half, bottom_half)).drop_duplicates()
print("mentions complete")

print("building mention neighbor dict\n")
neighbors = {}
neighbors_df = mentions.groupby("source")["target"].unique().reset_index()

for _, row in tqdm(neighbors_df.iterrows()):
    neighbors[row["source"]] = list(row["target"])

with open(paths["mentions"]["adjacency_list"], "w") as fout:
    json.dump(neighbors, fout)
print("mention neighbor dict complete")


# list of all frame names
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]

# public tweet metadata like ideology scores, engagement, etc.
meta = (pd.read_csv(paths["public"]["metadata"], sep="\t", dtype={"id_str": str})
          .drop(all_frame_list + ["Unnamed: 0", "Hero", "Threat", "Victim"], 
                axis="columns"))
meta_subset = pd.merge(filtered_tweets[["id_str"]], meta, how="left", on="id_str")
meta_subset["id_str"] = meta_subset["id_str"].astype(str)


def get_unique_mentions(user: str,
                        mentions: pd.DataFrame,
                        name_to_id_map: Dict,
                        id_to_name_map: Dict) -> Optional[Tuple[int, List[str]]]:
    user_id = ts.get_user_id(user, user_id_map)

    # if we have the user in our map we continue doing stuff
    if user in name_to_id_map:
        user_id = name_to_id_map[user]
        user_mentions = mentions[(mentions["uid1"] == user_id) | (mentions["uid2"] == user_id)]
        unique_mentions = user_mentions.shape[0]

        neighbor_ids = np.hstack((user_mentions["uid1"].values,
                                 user_mentions["uid2"].values))
        
        neighbor_ids = neighbor_ids[neighbor_ids != user_id]
        
        neighbors = []
        for id in neighbor_ids:

            # skip the self in making this list
            if id in id_to_name_map:
                name = id_to_name_map[id]

            # again we might not have the user in our map
                neighbors.append(name)

        return (unique_mentions, neighbors)

    else:
         return None


def get_neighbors(user: str,
                  mention_neighbors: dict,
                  name_to_id_map: Dict,
                  id_to_name_map: Dict) -> Optional[Tuple[int, List[str]]]:
    user_id = ts.get_user_id(user, user_id_map)

    # if we have the user in our map we continue doing stuff
    if user in name_to_id_map:

        user_id = name_to_id_map[user]
        if user_id in mention_neighbors:
            all_neighbors = mention_neighbors[user_id]
            unique_mentions = len(all_neighbors)

            neighbors = []
            for id in all_neighbors:

                # skip the self in making this list
                if id in id_to_name_map:
                    name = id_to_name_map[id]

                # again we might not have the user in our map
                    neighbors.append(name)

                return (unique_mentions, neighbors)
            else:
                return None

    else:
         return None



feature_rows = []
mention_neighbors = {}
for user in tqdm(filtered_tweets["screen_name"]):
    user_features = {}

    # convert user screen name to user id number
    lookup = get_neighbors(user, neighbors, name_to_id, id_to_name)

    if lookup:
        
        unique_mentions, targets = lookup
        user_features["log_unique_mentions"] = np.log(unique_mentions + 1)

        # add the tweet id to the row in case we need to merge later
        user_features["screen_name"] = user
        feature_rows.append(user_features)

        # add the network neighbors to a dictionary for later ;)
        mention_neighbors[user] = targets

# get screen names into the metadata df
names = filtered_tweets[["id_str", "screen_name"]]
meta_subset = pd.merge(meta_subset, names, on="id_str", how="left")
features_df = pd.merge(meta_subset, pd.DataFrame(feature_rows), how="left", on="screen_name")
features_df.to_csv(paths["regression"]["features"], sep="\t", index=False)

with open(paths["mentions"]["neighbors"], "w") as neighbors_out:
    json.dump(mention_neighbors, neighbors_out)
