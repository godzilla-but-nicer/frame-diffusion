# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import reduce
from itertools import permutations
from glob import glob

import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")
# %%
dyad_df = pd.read_csv("data/edge_lists/in_sample_dyads.tsv", sep="\t").drop_duplicates()
# %%
kinds, counts = np.unique(dyad_df["kind"], return_counts=True)


fig, ax = plt.subplots()
ax.bar(kinds, counts)
xticks = ax.get_xticks()

for x_pos, y_count in zip(xticks, counts):
    ax.text(x_pos, y_count, s=f"{y_count}", ha="center", va="bottom")

ax.set_xlabel("Dyad Kind")
ax.set_ylabel("Number in sample")
plt.show()
# %% [markdown]

# By a factor of five we have more replies than quotes, the second most common
# type of interaction. This is at least in part due to the fact that we did yet
# collect pure retweets from the decahose tweets. I suspecty that replies are
# also just way more common than the other two but we can double check once we
# get all of the relevant decahose data.
#
# ## Replies
#
# Let's break down whats going on in the replies
# %%
groups = dyad_df["source_group"].unique()
counts = np.zeros((len(groups), len(groups)))
replies_only = dyad_df["kind"] == "reply"

for s, source_group in enumerate(groups):
    for t, target_group in enumerate(groups):
        print(f"{source_group} -> {target_group}")
        source_filter = dyad_df["source_group"] == source_group
        target_filter = dyad_df["sample"].str.contains(target_group)  # sample column is separated into public_2018 and public_2019
        counts[s, t] = dyad_df[source_filter & target_filter & replies_only].shape[0]

group_normalizer = np.sum(counts, axis=1) + 1
probs = counts / group_normalizer

fig, ax = plt.subplots()

ax.imshow(counts, cmap="Greens")

for x in range(3):
    for y in range(3):
        ax.text(x, y, f"{counts[y, x]:.0f}", ha="center", va="center", size=18)

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])

ax.set_yticklabels(groups)
ax.set_xticklabels(groups)

ax.set_ylabel("Replying user in group")
ax.set_xlabel("Replied to group")

# %% [markdown]
#
# Ok so we can see that journalists most often reply to other journalists while
# occasionally replying to the public. We have no replies from congress so I
# dont know why i included that at all. Interestingly the public does not most
# often reply to itself despite being by far the largest class. This could be
# because of how sparse our coverage of this group is? Additionally we see far
# more replies to congress than to journalists, could this indicate that on
# twitter members of the public are bypassing the traditional access point to
# power and going straight to powerful people?
# 
# ## how often are replies reframing a conversation?
#
# %%
# let's just do the public for now
public_frames = pd.read_csv("data/binary_frames/predicted_frames.tsv", sep="\t")

replies_only = dyad_df["kind"] == "reply"
from_public = dyad_df["source_group"] == "public"
to_congress = dyad_df["sample"] == "congress"

replies_to_congress = dyad_df[replies_only & from_public & to_congress]

with open("workflow/config.json", "r") as config_file:
    config = json.loads(config_file.read())

generic_frames = config["frames"]["generic"].copy()
generic_frames.append("id_str")
public_generic_frames = public_frames[generic_frames]

congress_frames = pd.read_csv("data/binary_frames/congress/congress_generic.tsv", sep="\t")

reply_frames = 
# %%
