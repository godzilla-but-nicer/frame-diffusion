# %%
import os
if not os.getcwd().split("/")[-1] == "frame-diffusion":
    os.chdir("..")

import json as json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
from scipy.stats import entropy

from frame_stats import bootstrap_ci, draw_frame_frequencies
import data_selector as ds


with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())

# %%
# load data
tweets = pd.read_csv(paths["journalists"]["full_tweets"], sep="\t")
journos = pd.read_csv(paths["journalists"]["metadata"], sep="\t")
journos["screen_name"] = journos["username"]
tweets_metadata = pd.merge(tweets, journos, on="screen_name").drop("username", axis="columns")

frame_dfs = []
for frame_type in paths["journalists"]["frames"].keys():
    df = pd.read_csv(paths["journalists"]["frames"][frame_type], sep="\t").drop("text", axis="columns")
    frame_dfs.append(df)

frames = reduce(lambda l, r: pd.merge(l, r, on="id_str"), frame_dfs).fillna(0)

tweet_frames = pd.merge(tweets_metadata, frames, on="id_str")

bias = pd.read_csv(paths["journalists"]["mbfc"], sep="\t")

# %%
