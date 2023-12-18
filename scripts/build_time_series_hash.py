# %%
import numpy as np
import pandas as pd
import pickle
import json

from datetime import datetime as dt
from datetime import timedelta

import frame_stats as fs
import frame_stats.time_series as ts

import sys
import os

from tqdm import tqdm

if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

# config has frame names
print("loading config and paths")
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# paths to data files ( need to fix this)
with open("workflow/sample_paths.json", "r") as pf:
    paths = json.loads(pf.read())
print("config and paths loaded")


# load all of the frames and tweet time stamps etc.
print("loading tweets")
f = pd.read_csv(paths["all_frames"], sep="\t")
filtered_tweets = fs.filter_users_by_activity(f, 10)
print("tweets loaded")

# FOR TESTING
filtered_tweets = filtered_tweets.sample(frac=1)
# %%
user_time_series = {}
for user in tqdm(filtered_tweets["screen_name"].unique()):
    user_ts = ts.construct_frame_time_series(user, filtered_tweets, "1D", config)
    user_time_series[user] = user_ts


with open(paths["user_time_series"], "wb") as fout:
    pickle.dump(user_time_series, fout)
# %%
