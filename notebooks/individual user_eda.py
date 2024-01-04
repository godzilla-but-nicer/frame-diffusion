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
# f = fs.load_all_frames(paths, config)
# f.to_csv(paths["all_frames"], sep="\t", index=False)

# then we can load them instead of building the df
f = pd.read_csv("data/binary_frames/all_frames.tsv", sep="\t")
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
    tsdf = fs.time_series.construct_frame_time_series(user, filtered_tweets, "1D", config)
    ax[i].plot(tsdf["Economic"])
    ax[i].set_ylabel("Num. Tweets")
    ax[i].set_title("@" + user)

plt.tight_layout()
plt.show()
# %% [markdown]
#
# Ok lets just try plotting distributions and some summary statistics for
# frame usage at the user-level. So we'll get like total number of each frame
# used by each user
#
# %%
frame_usage_sums = filtered_tweets.groupby("screen_name").sum()
frame = "Threat: Public Order"

import seaborn as sns
plt.figure(dpi=300)
sns.histplot(frame_usage_sums, x=frame, discrete=True, log_scale=(False, True))
plt.ylabel("Number of Users")
plt.xlabel(f"Total {frame} tweets")

plt.savefig(f"plots/user_level/{frame}_hist.png")

# %%
from scipy.stats import entropy

entropies = []
for _, row in frame_usage_sums.drop("id_str", axis="columns").iterrows():
    user_dist = row.values / np.sum(row.values)
    entropies.append(entropy(user_dist, base=2))


# %%
plt.figure(dpi=300)
plt.hist(entropies, bins=50)
plt.axvline(np.log2(27), ls="--", c="k", label="Theoretical Bound")
plt.xlabel("User Frame Entropy (bit)")
plt.ylabel("Number of users")
plt.legend()
plt.savefig(f"plots/user_level/entropy.png")
# %%
import pickle

with open("data/regression/self_influence_pairs.pkl", "rb") as pkl_file:
    self_pairs = pickle.load(pkl_file)

self_users = set([])
for entry in tqdm(self_pairs):
    self_users.add(entry["t+1"]["screen_name"])
# %%
self_users_list = list(self_users)
in_sample = pd.DataFrame({"screen_name": self_users_list, "in_self_sample": True})

all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]
filtered_tweets_sample_labels = pd.merge(filtered_tweets, in_sample, how="left", on="screen_name").fillna({"in_self_sample": False})

user_activity = filtered_tweets_sample_labels.drop(["time_stamp", "id_str", "group"], axis="columns").groupby(["screen_name", "in_self_sample"]).sum()
# %%
frame = "Threat: Jobs"
plt.figure(dpi=300)
sns.distplot(user_activity.reset_index(), x=frame, hue="in_self_sample")
plt.savefig("plots/user_level/self_sample.png")
# %%
