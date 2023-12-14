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
import pickle

# we dont want to be working in notebooks/ for pathing reasons
import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

# config has frame names
print("loading config and paths")
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# paths to data files ( need to fix this)
with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())
print("config and paths loaded")

# connect screen names and user ids. needed to work with mention network
print("loading id map")
user_id_map = pd.read_csv(paths["public"]["user_id_map"], sep="\t",
                          dtype={"screen_name": str, "user_id": str})
print("id map loaded")


# load all of the frames and tweet time stamps etc.
print("loading tweets")
f = pd.read_csv(paths["all_frames"], sep="\t")
filtered_tweets = fs.filter_users_by_activity(f, 10)
print("tweets loaded")

# FOR TESTING
filtered_tweets = filtered_tweets.sample(frac=1)

# list of all frame names
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]

# load the features dataset
print("loading features")
features = pd.read_csv(paths["regression"]["features"], sep="\t")
features["id_str"] = features["id_str"].astype(str)
features = features.drop_duplicates()
print("features loaded")

# adjacency list for neighbor lookup
print("loading adjacency list")
with open(paths["mentions"]["adjacency_list"], "r") as fout:
    mention_neighbors = json.loads(fout.read())
print("adjacency list loaded")
# %% [markdown]
# Ok now we need to built the treatment pairs, Basically for each tweet we look
# for frames in the previous day. This is the sort of self exposure treatment.
# This is what we're calling `all_frame_pairs` a tweet and the frames from the day before
# %%
all_frame_pairs = []
print("gathering self-influence pairs \n\n")
for user in tqdm(filtered_tweets["screen_name"].unique()):
    try:
        user_pairs = ci.construct_tweet_self_influence_pairs(user,
                                                             filtered_tweets,
                                                             "1D",
                                                             config)
    except:
        continue

    if user_pairs:
        all_frame_pairs.extend(user_pairs)

all_frame_pairs

with open("self-influence-backup-pairs.pkl", "w") as fout:
    pickle.dump(all_frame_pairs, fout)
# %%
print("building self-influence dfs \n\n")
regression_dfs = {}
for frame in tqdm(all_frame_list):

    rows = []

    for pair in all_frame_pairs:
        exposure = pair["t"][frame]
        cue = pair["t+1"][frame]
        id = pair["t+1"]["id_str"]

        row = {}
        row["cue"] = cue
        row["id_str"] = str(id)
        row["exposure"] = exposure

        rows.append(row)

    pairs_df = pd.DataFrame(rows)

    regression_dfs[frame] = pd.merge(pairs_df, features, on="id_str", how="left")


# %%
endog_col = "cue"
exog_cols = ["exposure", 
              "is_quote_status", "is_reply", 
              "log_chars", "log_favorites", "log_retweets",
              "is_verified", "log_followers", "log_following", "log_statuses", "ideology", "log_unique_mentions"]

print("running self-influence regressions")
for frame in tqdm(all_frame_list):

    clean_frame_df = regression_dfs[frame].dropna(subset=[endog_col] + exog_cols)

    response = clean_frame_df[endog_col]
    predictors = sm.add_constant(clean_frame_df[exog_cols])

    model = sm.Logit(endog=response, exog=predictors)

    try:
        result = model.fit()
    except:
        print(f"no results for frame: {frame}")
        result = None
        continue

    header_suffix = f"{frame.lower().replace(' ', '_')}_header.csv"
    with open(paths["regression"]["self_output"] + header_suffix, "w") as f_head:
        f_head.write(result.summary().tables[0].as_csv())

    params_suffix = f"{frame.lower().replace(' ', '_')}_table.csv"
    with open(paths["regression"]["self_output"] + params_suffix, "w") as f_body:
        f_body.write(result.summary().tables[1].as_csv())

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
        try:
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

        except:
            continue

        if frame_pairs:
            all_frame_pairs.extend(frame_pairs)

all_frame_pairs
with open("alter-influence-backup-pairs.pkl", "w") as fout:
    pickle.dump(all_frame_pairs, fout)
# %%
print("building alter-influence dfs")
regression_dfs = {}
for frame in tqdm(all_frame_list):

    rows = []

    for pair in all_frame_pairs:
        exposure = pair["t"][frame]
        cue = pair["t+1"][frame]
        id = pair["t+1"]["id_str"]

        row = {}
        row["cue"] = cue
        row["id_str"] = str(id)
        row["exposure"] = exposure

        rows.append(row)

    pairs_df = pd.DataFrame(rows)

    regression_dfs[frame] = pd.merge(pairs_df, features, on="id_str", how="left").dropna()
# %%

print("running alter-influence regressions")
endog_col = "cue"
exog_cols = ["exposure", 
              "is_quote_status", "is_reply", 
              "log_chars", "log_favorites", "log_retweets",
              "is_verified", "log_followers", "log_following", "log_statuses", "ideology", "log_unique_mentions"]


for frame in tqdm(all_frame_list):
    response = regression_dfs[frame][endog_col]
    predictors = sm.add_constant(regression_dfs[frame][exog_cols])

    model = sm.Logit(endog=response, exog=predictors)

    try:
        result = model.fit()
    except:
        print(f"no results for frame: {frame}")
        continue

    header_suffix = f"{frame.lower().replace(' ', '_')}_header.csv"
    with open(paths["regression"]["alter_output"] + header_suffix, "w") as f_head:
        f_head.write(result.summary().tables[0].as_csv())

    params_suffix = f"{frame.lower().replace(' ', '_')}_table.csv"
    with open(paths["regression"]["alter_output"] + params_suffix, "w") as f_body:
        f_body.write(result.summary().tables[1].as_csv())

# %%
