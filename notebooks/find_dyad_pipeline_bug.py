# %%
import pandas as pd
import json
import numpy as np


import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

# first let's see if the dyad file has frame info
with open("data/down_sample/edge_lists/full_dyads.json") as json_file:
    dyads = json.loads(json_file.read())

with open("workflow/config.json") as config_file:
    config = json.loads(config_file.read())

print(len(dyads))

# %%
print(dyads[0])

# %%
print(json.dumps(dyads[0], indent = 4))
# %% [markdown]
# ## No frames in the dyad file!!
#
# Let's try the in sample dyads tsv file. Can we match the ids to the frames?
# %%
dyad_df = pd.read_csv("data/down_sample/edge_lists/in_sample_dyads.tsv", sep="\t")
frame_df = pd.read_csv("data/down_sample/binary_frames/all_group_frames.tsv", sep="\t").drop(["Unnamed: 0", "Victim", "Hero", "Threat", "text"], axis="columns")

# make a version of frame df with id str renamed tweet id for merging
# also one with target_id
source_frames = (frame_df
                 .add_prefix("s_")
                 .rename({"s_id_str": "tweet_id"}, axis="columns"))
target_frames = (frame_df
                 .add_prefix("t_")
                 .rename({"t_id_str": "target_id"}, axis="columns"))

dyads_source = pd.merge(dyad_df, source_frames, on="tweet_id")
dyads_target = pd.merge(dyad_df, target_frames, on="target_id")
dyads_framed = pd.merge(dyads_source, dyads_target, on=["tweet_id", "target_id", "kind", "source_group", "sample"])
dyads_framed = pd.concat([dyads_framed, dyads_target[dyads_target["kind"] == "retweet"]])

# %% [markdown] Ok so the problem is clearly about inconsistant use of str/int
# to id the tweets very annoying. Can we replace all of the matching with
# pandas? idk
#
# Ok heres what I will try. First, just fixing the gather dyads script to use
# the correct types. Then we'll go from there
#
# Better solution might be to make a pair of dataframes as we've done above for
# sources and targets and use those for matching

# %%
