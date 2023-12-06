# %% [markdown]
# # Logistic Regression of frame influence
#
# This notebook contains most of the code for running the regressions we will
# use to estimate the effects of frame exposure on frame usage. We are
# focusing on the general public in this work since we have the most complete
# data for this group.
#
# In the first cell we are loading a bunch of libraries and data. I will note
# in comments what these things are used for where appropriate.
# %%
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

# we dont want to be working in notebooks/ for pathing reasons
import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

# config has frame names
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# paths to data files ( need to fix this)
with open("workflow/sample_paths.json", "r") as pf:
    paths = json.loads(pf.read())

# connect screen names and user ids. needed to work with mention network
user_id_map = pd.read_csv("data/down_sample/user_id_map.tsv", sep="\t",
                          dtype={"screen_name": str, "user_id": str})

# the mention network itself
mentions = pd.read_csv("data/down_sample/edge_lists/in_sample_mentions.tsv", sep="\t",
                       dtype={"uid1": str, "uid2": str,
                              "1to2freq": int, "2to1freq": int})

# load the tweet json so we can grab additional details
with open("data/down_sample/immigration_tweets/tweets_by_id.json", "r") as catalog_raw:
    catalog = json.loads(catalog_raw.read())

# load all of the frames and tweet time stamps etc.
f = pd.read_csv("data/down_sample/binary_frames/all_frames.tsv", sep="\t")
filtered_tweets = fs.filter_users_by_activity(f, 10)

# FOR TESTING
filtered_tweets = filtered_tweets.sample(frac=0.2)

# list of all frame names
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]

# public tweet metadata like ideology scores, engagement, etc.
meta = pd.read_csv("data/user_info/full_datasheet.tsv", sep="\t").drop(all_frame_list + ["Unnamed: 0", "Hero", "Threat", "Victim"], axis="columns")
meta_subset = pd.merge(filtered_tweets[["id_str"]], meta, how="left", on="id_str")
meta_subset["id_str"] = meta_subset["id_str"].astype(str)
# %% [markdown]
# ## Information from mention network
#
# We want to use degree in the mention network as a feature in our regressions.
# We're also going to need a list of neighbors for each user to get the
# alter time series.
# %%
# creating a features dataframe for each tweet
# I think with the full datasheet I can actually just skip all of this
# user_id_map_dict = {x["screen_name"].lower(): x["user_id"] for _, x in user_id_map.iterrows()}
# user_ids = filtered_tweets["screen_name"].map(user_id_map_dict)

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

# %% [markdown]
# Ok now we need to built the treatment pairs, Basically for each tweet we look
# for frames in the previous day. This is the sort of self exposure treatment.
# This is what we're calling `all_frame_pairs` a tweet and the frames from the day before
# %%
all_frame_pairs = []

for user in filtered_tweets["screen_name"].unique():
    user_pairs = ci.construct_tweet_self_influence_pairs(user,
                                                         filtered_tweets,
                                                         "1D",
                                                         config)

    if user_pairs:
        all_frame_pairs.extend(user_pairs)

all_frame_pairs
# %%
def build_regression_dfs(frame_list: List[str],
                         frame_pairs: List[Dict],
                         features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    regression_dfs = {}
    for frame in tqdm(frame_list):
        rows = []
        for pair in frame_pairs:
            exposure = pair["t"][frame]
            cue = pair["t+1"][frame]
            id = pair["t+1"]["id_str"]

            row = {}
            row["cue"] = cue
            row["id_str"] = str(id)
            row["exposure"] = exposure

            rows.append(row)

        regression_dfs[frame] = pd.merge(pd.DataFrame(rows), features, on="id_str").dropna()

    return regression_dfs

regression_dfs = build_regression_dfs(all_frame_list, all_frame_pairs, features_df)
# %%
endog_col = "cue"
exog_cols = ["exposure", 
              "is_quote_status", "is_reply", 
              "log_chars", "log_favorites", "log_retweets",
              "is_verified", "log_followers", "log_following", "log_statuses", "ideology", "log_unique_mentions"]

with open("data/down_sample/regression_output/self_influence.txt", "w") as f:
    pass

for frame in tqdm(all_frame_list):
    response = regression_dfs[frame][endog_col]
    predictors = sm.add_constant(regression_dfs[frame][exog_cols])

    model = sm.Logit(endog=response, exog=predictors)

    try:
        result = model.fit()
    except:
        print(f"no results for frame: {frame}")
        continue

    with open("data/down_sample/regression_output/self_influence.txt", "a") as f:
        f.write("\n\n\n")
        f.write(f"{frame} -----------------")
        f.write("\n")
        f.write(result.summary().as_text())
# %% [markdown]
#
# Now we have to do the alter influence regressions. The functions should be
# partially done and we just have to put things into shape
#
# %%
all_frame_pairs = []


# ok so this is going to be slow as hell
print("Checking tweet pairs for alter influence")
for user in tqdm(filtered_tweets["screen_name"].unique()):

    # we gathered a list of neighbors in an earlier cell ;)
    # now we just have to collect the time series here to pass into the function
    # kind of ugly honestly
    alter_ts_list = []
    if user in mention_neighbors:

        for alter in mention_neighbors[user]:
            alter_ts_list.append(ts.construct_frame_time_series(filtered_tweets,
                                                                alter,
                                                                "1D",
                                                                config))

        frame_pairs = ci.construct_tweet_alter_influence_pairs(user,
                                                               alter_ts_list,
                                                               filtered_tweets,
                                                               "1D",
                                                               config)

        if frame_pairs:
            all_frame_pairs.extend(frame_pairs)

all_frame_pairs
# %%
regression_dfs = build_regression_dfs(all_frame_list, all_frame_pairs, features_df)

endog_col = "cue"
exog_cols = ["exposure", 
              "is_quote_status", "is_reply", 
              "log_chars", "log_favorites", "log_retweets",
              "is_verified", "log_followers", "log_following", "log_statuses", "ideology", "log_unique_mentions"]

with open("data/down_sample/regression_output/alter_influence.txt", "w") as f:
    pass

for frame in tqdm(all_frame_list):
    response = regression_dfs[frame][endog_col]
    predictors = sm.add_constant(regression_dfs[frame][exog_cols])

    model = sm.Logit(endog=response, exog=predictors)

    try:
        result = model.fit()
    except:
        print(f"no results for frame: {frame}")
        continue

    with open("data/down_sample/regression_output/alter_influence.txt", "a") as f:
        f.write("\n\n\n")
        f.write(f"{frame} -----------------")
        f.write("\n")
        f.write(result.summary().as_text())

# %%
