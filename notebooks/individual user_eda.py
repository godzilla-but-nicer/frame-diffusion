# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm
from typing import Dict, List
import frame_stats as fs
import frame_stats.time_series as ts
import frame_stats.causal_inferrence as ci

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())

# %%
# the first time we run this let's just save all the frames
f = fs.load_all_frames(paths, config)
f.to_csv(paths["all_frames"], sep="\t", index=False)

# then we can load them instead of building the df
# f = pd.read_csv("data/down_sample/binary_frames/all_frames.tsv", sep="\t")
# %%
filtered_tweets = fs.filter_users_by_activity(f, 10)
# %%
# ok let's try and look at the time series of the most active users to see
# how active they are really
most_active_users = filtered_tweets.groupby("screen_name").count()["id_str"].reset_index().sort_values("id_str", ascending=False)["screen_name"]
top_user = list(most_active_users)[3]
# %%
fig, ax = plt.subplots(nrows=5, figsize=(5, 10))

for i, user in enumerate(list(most_active_users)[:5]):
    tsdf = fs.time_series.construct_frame_time_series(filtered_tweets, user, "1D", config).set_index("time_stamp")
    ax[i].plot(np.diff(tsdf.sum(axis=1)))
    ax[i].set_ylabel("Num. Tweets")
    ax[i].set_title("@" + user)

plt.tight_layout()
plt.show()
# %% [markdown]
#
# Ok lets just try this thing in the obvious way. We'll pick a user, grab
# their neighbors from the mention network, and do the regression.
#
# we'll go with the top user. In order to get their mention network neighbors
# we will need to convert screen name to user id than find the out links from
# that id in the mention network.
#
# I'm going to make a big lookup table for screen name to user id in another
# script and just load it here.

# %%
user_id_map = pd.read_csv("data/down_sample/user_id_map.tsv", sep="\t",
                          dtype={"screen_name": str, "user_id": str})

# %% [markdown]
# Ok now I guess let's just try and build some of these things. We'll need
# the mention network and the time series for basically every user in our
# dataset.
# 

# %%
mentions = pd.read_csv("data/down_sample/edge_lists/in_sample_mentions.tsv", sep="\t",
                       dtype={"uid1": str, "uid2": str,
                              "1to2freq": int, "2to1freq": int})

def longer_mention_subset(mention_subset: pd.DataFrame) -> pd.DataFrame:
    first_half_cols = ["uid1", "uid2", "1to2freq"]
    first_half = mention_subset[first_half_cols]
    first_half["source"] = first_half["uid1"]
    first_half["target"] = first_half["uid2"]
    first_half["weight"] = first_half["1to2freq"]
    first_half = first_half[["source", "target", "weight"]]

    second_half_cols = ["uid1", "uid2", "2to1freq"]
    second_half = mention_subset[second_half_cols]
    second_half["source"] = second_half["uid2"]
    second_half["target"] = second_half["uid1"]
    second_half["weight"] = second_half["2to1freq"]
    second_half = second_half[["source", "target", "weight"]]

    return pd.concat((first_half, second_half))

def get_screen_name(user_id, id_map: pd.DataFrame) -> str:
    return id_map[id_map["user_id"] == user_id]["screen_name"].values[0]


def get_user_id(screen_name, id_map: pd.DataFrame) -> str:
    return id_map[id_map["screen_name"] == screen_name]["user_id"].values[0]

def get_mentioned_screen_names(screen_name: str,
                               mentions: pd.DataFrame,
                               id_map: pd.DataFrame) -> dict:
    
    user_id = get_user_id(screen_name, id_map)
    user_ego = mentions[(mentions["uid1"] == user_id) | (mentions["uid2"] == user_id)]
    user_ego_longer = longer_mention_subset(user_ego)

    user_is_source = user_ego_longer[user_ego_longer["source"] == user_id]
    
    influencers = []
    for i, alter in user_is_source.iterrows():
        alter_screen_name = get_screen_name(alter["target"], id_map)
        alter_weight = alter["weight"]

        influencers.append({"screen_name": alter_screen_name, "weight": alter_weight})

    return influencers

get_mentioned_screen_names(top_user, mentions, user_id_map)

def get_influencer_time_series(focal_user: str,
                               tweets: pd.DataFrame,
                               config: Dict,
                               mentions: pd.DataFrame,
                               id_map: pd.DataFrame) -> List:
    influencers = get_mentioned_screen_names(focal_user, mentions, id_map)

    influencer_ts_list = []

    for influencer in influencers:

        influencer_ts_info = {}
        influencer_ts_df = fs.construct_frame_time_series(tweets, influencer["screen_name"], "1D", config).set_index("time_stamp")
        
        influencer_ts_info["screen_name"] = influencer["screen_name"]
        influencer_ts_info["weight"] = influencer["weight"]
        influencer_ts_info["time_series"] = influencer_ts_df

        influencer_ts_list.append(influencer_ts_info)
    
    return influencer_ts_list

top_user_influencers = get_influencer_time_series(top_user, filtered_tweets, config, mentions, user_id_map)
# %%
fig, ax = plt.subplots(nrows=len(top_user_influencers) + 1, figsize=(6,10))

top_user_ts = fs.construct_frame_time_series(filtered_tweets, top_user, "1D", config).set_index("time_stamp")

ax[0].plot(top_user_ts["Policy Prescription and Evaluation"])
ax[0].set_title("@" + top_user)

for i, influencer in enumerate(top_user_influencers):
    ts = influencer["time_series"]
    ax[i+1].plot(ts["Policy Prescription and Evaluation"])
    ax[i+1].set_title("@" + influencer["screen_name"])

plt.tight_layout()
# %% [markdown]
# Ok so we need to get pairs of tweets for a user that happen on consecutive days
# really I guess we need consecutive days for which the user has cued tweets both days

# %%
day_pairs = []
one_day = pd.Timedelta(1, "day")

for t, day in top_user_ts.iterrows():
    
    if (top_user_ts.loc[t].sum() > 0 and
        top_user_ts.loc[t + one_day].sum() > 0):
        
        pair = {}
        pair["t"] = top_user_ts.loc[t]
        pair["t+1"] = top_user_ts.loc[t + one_day]

        day_pairs.append(day)

filtered_tweets
# %% [markdown]
#
# Ok so we need to figure out the propensity score matching. We're really using
# tweets in our analysis rather than users. each treated/untreated pair is a
# pair of tweets but users will appear more than once in this list. So we want
# to pull out whatever user info we have access to as well as info for the
# tweets (time stamp, metrics?)
#
# Let's first just look at what info i have access to. we will load some tweet json

# %%
import json
import gzip


print("Catalogging Public tweets")
for tweet_json in open(paths["journalists"]["tweet_json"]):
    tweets = json.loads(tweet_json)
    for tweet in tweets:
        print(json.dumps(tweet, indent=2))
        break

# %%
dir(fs)
# %%
