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
with open("workflow/sample_paths.json", "r") as pf:
    paths = json.loads(pf.read())

# connect screen names and user ids. needed to work with mention network
user_id_map = pd.read_csv(paths["mentions"]["id_map"], sep="\t",
                          dtype={"screen_name": str, "user_id": str})

# the mention network itself
mentions = pd.read_csv(paths["mentions"]["network"], sep="\t",
                       dtype={"uid1": str, "uid2": str,
                              "1to2freq": int, "2to1freq": int})

# load the tweet json so we can grab additional details
with open(paths["tweet_catalog"], "r") as catalog_raw:
    catalog = json.loads(catalog_raw.read())

# load all of the frames and tweet time stamps etc.
f = pd.read_csv(paths["all_frames"], sep="\t")
filtered_tweets = fs.filter_users_by_activity(f, 10)

# FOR TESTING
filtered_tweets = filtered_tweets.sample(frac=1)

# list of all frame names
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]

# public tweet metadata like ideology scores, engagement, etc.
meta = (pd.read_csv(paths["public"]["metadata"], sep="\t")
          .drop(all_frame_list + ["Unnamed: 0", "Hero", "Threat", "Victim"], 
                axis="columns"))
meta_subset = pd.merge(filtered_tweets[["id_str"]], meta, how="left", on="id_str")
meta_subset["id_str"] = meta_subset["id_str"].astype(str)


def get_unique_mentions(user: str,
                        mentions: pd.DataFrame,
                        user_id_map: pd.DataFrame) -> Optional[Tuple[int, List[str]]]:
    user_id = ts.get_user_id(user, user_id_map)

    # if we have the user in our map we continue doing stuff
    if user_id:
        user_mentions = mentions[(mentions["uid1"] == user_id) | (mentions["uid2"] == user_id)]
        unique_mentions = user_mentions.shape[0]

        neighbor_ids = np.hstack((user_mentions["uid1"].values,
                                 user_mentions["uid2"].values))
        
        neighbors = []
        for id in neighbor_ids:

            # skip the self in making this list
            if id != user_id:
                name = ts.get_screen_name(id, user_id_map)

                # again we might not have the user in our map
                if name:
                    neighbors.append(name)

        return (unique_mentions, neighbors)

    else:
         return None


feature_rows = []
mention_neighbors = {}
for i, tweet in tqdm(filtered_tweets.iterrows()):
    tweet_features = {}

    # convert user screen name to user id number
    if tweet["screen_name"] in user_id_map["screen_name"].values:

        lookup = get_unique_mentions(tweet["screen_name"], mentions, user_id_map)

        if lookup:
            unique_mentions, neighbors = lookup
            tweet_features["log_unique_mentions"] = np.log(unique_mentions + 1)

            # add the tweet id to the row in case we need to merge later
            tweet_features["id_str"] = str(tweet["id_str"])
            feature_rows.append(tweet_features)

            # add the network neighbors to a dictionary for later ;)
            mention_neighbors[tweet["screen_name"]] = neighbors


features_df = pd.merge(meta_subset, pd.DataFrame(feature_rows), how="left", on="id_str")
features_df.to_csv(paths["regression"]["features"], sep="\t", index=False)
