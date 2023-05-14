# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json as json
import textwrap
from scipy.stats import entropy

with open("../workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

np.random.seed(2133)
# %%
groups = ["congress", "journalists", "trump", "public"]
frame_types = ["generic", "specific", "narrative"]  # narrative should be in here too

tweets = {k: {} for k in groups}

for group in groups:
    for frame_type in frame_types:

        # all frames are in one file for the public
        if group != "public":
            predictions = pd.read_csv(f"../data/binary_frames/{group}/{group}_{frame_type}.tsv",
                                      sep="\t")
            tweets[group][frame_type] = predictions

        else:
            predictions = pd.read_csv(f"../data/binary_frames/predicted_frames.tsv",
                                      sep="\t")
            us_public_ids = pd.read_csv(f"../data/us_public_ids.tsv", sep="\t")
            predictions = pd.merge(us_public_ids, predictions,
                                   on="id_str", how="left")
            tweets[group] = predictions
            break

def sample_cooccurring_frames(df, pair):
    cooccurring = df[(df[frame_pair[0]] == 1) & (df[frame_pair[1]] == 1)]
    sample = cooccurring.sample(5)

    print(f"{frame_pair[0]} and {frame_pair[1]}")
    print("-"*70 + "\n")
    for sample_text in sample["text"]:
        lines = textwrap.wrap(f"{sample_text}", 
                                70, 
                                subsequent_indent="        ", 
                                initial_indent="--->  ")

        for line in lines:
            print(line)

# %% [markdown]
#
# What I want to do in this notebook is pull out a random handful of tweets
# where frames cooccur and actually read them to get a sense for what is behind
# the cooccurance.
#
#  ## Congress cooccurances
#
# I think in the congressional patterns we are really seeing a partisan divide
# where Dems are using particular sets of frames and Republicans are using
# others. Some of the pairs we'll look at ought to be obvious like Policy
# Prescriptions and Political Factors. Others will hopefully reveal themselves,
# like Economic and External Regulation.
#
# ### General frames
#
# I want to do two pairs listed above as well as Economic and Capacity and
# resources.

# %%
df = tweets["congress"]["generic"]

frame_pair = ("Economic", "Capacity and Resources")

sample_cooccurring_frames(df, frame_pair)
# %% [markdown]
#
# To me, none of these tweets are about economics in the way i was thinking of
# it as generally about people's ability to afford stuff with the exception of
# the third in that it mentions jobs. Instead we see more of a concern about
# how fenderal funds are spent and how that affects the capacity of the
# government to do stuff and/or the financial resources with which it would.
#  The third tweet appears to be about Capacity and Resources in that a house
#  is a resource.
# %%
frame_pair = ("External Regulation and Reputation", "Economic")
sample_cooccurring_frames(df, frame_pair)
# %% [markdown]
# 
# If we assume that diplomacy in general is External Reputation than I'm
# definitely seeing it here. Economics tends to appear mostly in the form of
# mentioning taxes. I think maybe the model is also picking up international
# aid as belonging to both frames and thus giving them both as present. This
# is, in my opinion, slightly different than cooccurance
# 
# %%
frame_pair = ("Political Factors and Implications",
              "Policy Prescription and Evaluation")
sample_cooccurring_frames(df, frame_pair)
# %% [markdown]
#
# The prescriptions here mostly seem implicit and I guess I don't feel equipped
# to define what i think "political factors" ought to mean until ive read the
# code book