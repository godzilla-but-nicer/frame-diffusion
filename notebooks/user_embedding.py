# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# %%
groups = ["congress", "journalists", "trump"]
frame_types = ["generic", "specific"]
all_frame_dfs = []

frames = {k: {} for k in groups}
for group in tqdm(groups):
    type_dfs = []
    if group != "public":
        for frame_type in frame_types:

            # all frames are in one file for the public
            # we have to merge a couple of different files in either case
            if group != "public":
                # predicted frames
                predictions = pd.read_csv(f"data/binary_frames/{group}/{group}_{frame_type}.tsv",
                                          sep="\t").drop("text", axis="columns")

                # time stamps for granger causality
                tweet_info = pd.read_csv(f"data/immigration_tweets/{group}.tsv",
                                         sep="\t")[["id_str", "time_stamp", "screen_name"]]

                tweet_info["id_str"] = tweet_info["id_str"].astype(str)
                predictions["id_str"] = predictions["id_str"].astype(str)

                # merge and add to list of dataframes to combine
                tweet_df = pd.merge(tweet_info, predictions,
                                    how="right", on="id_str")
                tweet_df["group"] = group
                type_dfs.append(tweet_df)

        all_frames_for_group = reduce(lambda l, r: pd.merge(l, r,
                                                            on=["id_str",
                                                                "time_stamp",
                                                                "screen_name",
                                                                "group"]),
                                      type_dfs)

    else:
        # frame predictions
        predictions = pd.read_csv(f"data/binary_frames/predicted_frames.tsv",
                                  sep="\t")
        tweet_info = pd.read_csv("data/us_public_ids.tsv", sep="\t")
        tweet_info["time_stamp"] = pd.to_datetime(tweet_info["time_stamp"])

        # tweet_info["time_stamp"] = datetime.strptime(tweet_info["time_stamp"],
        #                                              "%a %b %d %H:%M:%S +0000 %Y")

        all_frames_for_group = pd.merge(tweet_info, predictions, on="id_str")
        all_frames_for_group["group"] = "public"

    all_frame_dfs.append(all_frames_for_group)

    all_frames = pd.concat(all_frame_dfs)

# %%
from sklearn.decomposition import PCA
import scipy.linalg as sla

trump_frames = all_frames[all_frames["group"] == "trump"]
trump_frames.head()

# we will quantify the spread using the square root of the determinant of the
# covariance matrix. This is explained here: https://stats.stackexchange.com/questions/72228/measures-of-multidimensional-spread-or-variance

trump_frame_mat = (trump_frames
                   .loc[:, (trump_frames != 0).any(axis=0)]
                   .drop(["id_str", "time_stamp", "screen_name", "group"],
                         axis="columns")
                   .values)
trump_cov = np.cov(trump_frame_mat.T)
trump_spread = np.sqrt(sla.det(trump_cov))


def frame_spread(df: pd.DataFrame) -> float:

    frame_mat = (df
                 .loc[:, (df != 0).any(axis=0)]
                 .drop(["id_str", "time_stamp", "screen_name", "group"],
                       axis="columns")
                 .values)
    
    frame_cov = np.cov(frame_mat.T)

    return np.sqrt(np.abs(sla.det(frame_cov)))
    

# %%
import warnings
warnings.filterwarnings("error")

user_spreads = []
for i, user in enumerate(tqdm(all_frames["screen_name"].unique())):
    user_frames = all_frames[all_frames["screen_name"] == user]
    if user_frames.shape[0] < 10:
        continue

    try:
        user_spreads.append(frame_spread(user_frames))
    
    except RuntimeWarning as w:
        print(w)
        print(user_frames)
        break

    except ValueError as e:
        print(e)
        print(user_frames)
        break


plt.hist(user_spreads, bins=30, log=True)
plt.show()
# %%
