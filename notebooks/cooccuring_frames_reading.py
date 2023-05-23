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
# Mostly straightforward, we're talking about money and in most cases how the
# money should be allocated to deal with problems. Not so sure about the third
# tweet except I guess that a house is a resource.
# %%
frame_pair = ("External Regulation and Reputation", "Economic")
sample_cooccurring_frames(df, frame_pair)
# %% [markdown]
# 
# External appears mostly as discussion of diplomacy. Economics tends to appear
# mostly in the form of mentioning taxes. International aid is both in that aid
# is diplomatic money and money is economic.
# 
# %%
frame_pair = ("Political Factors and Implications",
              "Policy Prescription and Evaluation")
sample_cooccurring_frames(df, frame_pair)
# %% [markdown]
#
# Ok we have discussion of political parties and branches along with specific
# mentions of policy. seems right.
#
# ### Congress - Specific frames
#
# Ok so we only have one pair here and Victim: War is the lowest performing
# frame from the classifier. I don't expect much.

# %%
df = tweets["congress"]["specific"]

frame_pair = ("Victim: War", "Victim: Global Economy")
sample_cooccurring_frames(df, frame_pair)
# %% [markdown] 
# We've got conflict for sure but the codebook reserves this
# frame specifically for named conflicts so i think these are false positives.
# I'm actually not so sure about the Global Economy frame either here except i
# guess implicit in the last tweet about underlying causes of instability.
#
# ## Journalist cooccurances
#
# we have many of the same correlated pairs here, i suspect its all similar to
# congress.
# 
# ### Journalists - Generic
#
# %%
df = tweets["journalists"]["generic"]

frame_pair = ("Economic", "Capacity and Resources")
sample_cooccurring_frames(df, frame_pair)
# %% [markdown] 
# Actually pretty different than what congress was saying! Much
# less focus on specific policy and who is doing what. Not sure what tweet two
# has to do with capacity and resources.

# %%
frame_pair = ("Health and Safety", "Crime and Punishment")
sample_cooccurring_frames(df, frame_pair)
# %% [markdown]
#
# Variety of things here covering both specific crimes and how their victims
# were affected, criminal groups, detention centers. These are all slam dunk
# crime tweets often with just a word or two of healkth and safety in
# mentioning the victims. in the last case its just crime?
#

# %%
frame_pair = ("Fairness and Equality", "Cultural Identity")
sample_cooccurring_frames(df, frame_pair)

# %% [markdown]
#
# Overall pretty spot on, the first one addresses culture and fairness
# directly, the others compare how cultural and/or racial groups are treated.
# the last one is wild.
#

# %%
frame_pair = ("External Regulation and Reputation", "Economic")
sample_cooccurring_frames(df, frame_pair)
# %% [markdown]
# 
# more straightforwardly cuing both frames than among the congress people.
# In most cases I think cuing them seperately. The second tweet is probably the
# minimal example of this combination
#
# %%
frame_pair = ("Political Factors and Implications",
              "Policy Prescription and Evaluation")
sample_cooccurring_frames(df, frame_pair)

# %% [markdown]
#
# This is similar to how congress people tralked about these. Mentions of
# specific policy prescriptions and by whom.
#
# %%
df = tweets["journalists"]["specific"]

frame_pair = ("Victim: War", "Victim: Global Economy")
sample_cooccurring_frames(df, frame_pair)
# %%
#
# Again similar to the congress people but I blame it on the classifier. some
# of these are about the "global economy" in that they mention other countries
# economies, not really named conflicts in here.
#
# ## Trump cooccurances
#
# These are the fun ones!
# %%