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
from sys import argv

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

# paths to data files
with open("workflow/sample_paths.json", "r") as pf:
    paths = json.loads(pf.read())
print("config and paths loaded")

# load all of the frames and tweet time stamps etc.
print("loading tweets")
f = pd.read_csv(paths["all_frames"], sep="\t", dtype={"id_str": str})
filtered_tweets = fs.filter_users_by_activity(f, 1)
filtered_tweets["time_stamp"] = pd.to_datetime(f["time_stamp"])
print("tweets loaded")

# list of all frame names
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]

# load the features dataset
print("loading features")
features = pd.read_csv(paths["regression"]["features"], sep="\t", dtype={"id_str": str})
features = features.drop_duplicates()
print("features loaded")

# adjacency list for neighbor lookup
print("loading adjacency list")
with open(paths["mentions"]["adjacency_list"], "r") as fout:
    mention_neighbors = json.loads(fout.read())
print("adjacency list loaded")

print("loading user time series hash")
with open(paths["user_time_series"], "rb") as utspkl:
    user_time_series = pickle.load(utspkl)
print("loaded user time series hash")

# %%
# this time we want to do all pairs whether or not a user cued frames on 
# consecutive days. So basically I'm going to rewrite my functions using the
# time seires hashmap to work in this more general way

def get_user_regression_pairs(user: str,
                              user_tweets: pd.DataFrame,
                              frame_time_series: pd.DataFrame,
                              window_days: int) -> List[Dict]:
    
    user_pairs = []

    # relevant timedelatas
    window = pd.Timedelta(f"{window_days}D")
    oneday = pd.Timedelta("1D")

    for _, tweet in user_tweets.iterrows():

        new_pair = {}
        new_pair["t+1"] = tweet  # full tweet into into "t+1" slot

        # sum frames within the window
        posting_date = tweet["time_stamp"].date()
        frames_in_window = frame_time_series[posting_date - window : posting_date - oneday]
        total_window_frames = frames_in_window.sum()
        exposures = frames_in_window.astype(bool).astype(int)
        new_pair["t"] = exposures

        user_pairs.append(new_pair)
    
    return user_pairs

import warnings

if not argv[1] == "--skip-self":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        self_regression_pairs = []
        for user in tqdm(filtered_tweets["screen_name"].unique()):
            user_tweets = filtered_tweets[filtered_tweets["screen_name"] == user]
            try:
                self_regression_pairs.extend(get_user_regression_pairs(user,
                                                                       user_tweets,
                                                                       user_time_series[user],
                                                                       1))
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                continue

    with open(paths["regression"]["self_influence_pairs"], "wb") as fout:
        pickle.dump(self_regression_pairs, fout)
# %%
# same proceedure for the alter pairs
def get_alter_regression_pairs(user,
                               user_tweets,
                               time_series_hash,
                               alter_names,
                               window_days) -> List[Dict]:
    

    user_pairs = []

    # relevant timedelatas
    window = pd.Timedelta(f"{window_days}D")
    oneday = pd.Timedelta("1D")

    for _, tweet in user_tweets.iterrows():

        new_pair = {}
        new_pair["t+1"] = tweet  # full tweet into into "t+1" slot

        # sum frames within the window for each alter
        frame_array = np.zeros(time_series_hash[user].shape[1])
        for alter in alter_names:
            posting_date = tweet["time_stamp"].date()
            frames_in_window = time_series_hash[alter][posting_date - window : posting_date - oneday]
            total_window_frames = frames_in_window.sum()
            frame_array += total_window_frames.values
        
        frame_series = pd.Series(frame_array, index=time_series_hash[user].columns)
        exposures = frames_in_window.astype(bool).astype(int)
        new_pair["t"] = exposures

        user_pairs.append(new_pair)
    
    return user_pairs

import warnings

alter_regression_pairs = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for user in tqdm(filtered_tweets["screen_name"].unique()):
        user_tweets = filtered_tweets[filtered_tweets["screen_name"] == user]
        try:

            alter_regression_pairs.extend(get_alter_regression_pairs(user,
                                                                   user_tweets,
                                                                   user_time_series,
                                                                   mention_neighbors[user],
                                                                   1))
        except Exception as e:
            # print(f"{type(e).__name__}: {e}")
            continue


with open(paths["regression"]["alter_influence_pairs"], "wb") as fout:
    pickle.dump(alter_regression_pairs, fout)
# %%
