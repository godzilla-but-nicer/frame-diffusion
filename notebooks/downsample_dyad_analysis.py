# %%
import pandas as pd
import json
import numpy as np
from typing import List, Dict, Union


import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

# first let's see if the dyad file has frame info
with open("data/down_sample/edge_lists/full_dyads.json") as json_file:
    raw_dyads = json.loads(json_file.read())

dyads = [json.loads(dyad) for dyad in raw_dyads]

with open("workflow/config.json") as config_file:
    config = json.loads(config_file.read())

print(len(dyads))

# %% [markdown]

# ## Conditional Probabilities for frame occurance given interaction with frame
#
# We want to calculate for each frame, the probability of a frame being used
# in a sorce tweet given that it appears in a target tweet. We should probably
# separate this by quotes and replies.
#
# $$ P(f_i \in S | f_i \in T) $$
#
# where $S$ is the source tweet (the tweet replying or quoting) and $T$ is the
# target tweets (the tweet being replied to or quoted) and $f_i$ is a
# particular frame.

# %%
def frame_conditional_prob(frame: str,
                           dyads: List[Dict],
                           interaction=None) -> float:
    
    target_positive = 0
    source_and_target = 0
    for dyad in dyads:

        # some kind of relationship checking could go here

        # next we look at if frame is identified in target
        if dyad["target_frames"][frame] == 1:
            target_positive += 1

            if dyad["source_frames"][frame] == 1:
                source_and_target += 1

    # if we never observe a frame in the target we want nan
    if target_positive > 0:
        return source_and_target / target_positive
    else:
        return np.nan

def test_frame_conditional_prob():
    data = [{"source_frames": {"a": 1}, "target_frames": {"a": 0}},
            {"source_frames": {"a": 1}, "target_frames": {"a": 1.0}},
            {"source_frames": {"a": 0.0}, "target_frames": {"a": 1}}]
    
    print(frame_conditional_prob("a", data) == 0.5)
    assert frame_conditional_prob("a", data) == 0.5

test_frame_conditional_prob()
# %% [markdown]
#
# Ok so let's run it and see what we see
#

# %%
all_frames = np.hstack([config["frames"]["generic"],
                        config["frames"]["specific"],
                        config["frames"]["narrative"]])

probs = {}
for frame in all_frames:
    probs[frame] = frame_conditional_prob(frame, dyads)

print(json.dumps(probs, indent=2))
# %% [markdown]
#
# I'm not sure what exactly this tells us on its own at least with respect to
# our driving questions. so We probably want to add the functionality that
# discriminates by reply vs quote. I think we expect for replies to agree more
# than quotes. Let's start by rewriting our function to do this.
#

# %%
def frame_conditional_prob(frame: str,
                           dyads: List[Dict],
                           by_kind=True) -> Union[float, Dict]:

    if not by_kind:
        target_positive = 0
        source_and_target = 0
        for dyad in dyads:

            # next we look at if frame is identified in target
            if dyad["target_frames"][frame] == 1:
                target_positive += 1

                if dyad["source_frames"][frame] == 1:
                    source_and_target += 1

        # if we never observe a frame in the target we want nan
        if target_positive > 0:
            return source_and_target / target_positive
        else:
            return np.nan
    
    else:
        target_positive = {kind: 0 for kind in ["reply", "quote"]}
        source_and_target = {kind: 0 for kind in ["reply", "quote"]}

        # same proceedure but now we stick the numbers in a dict
        for dyad in dyads:
            
            if dyad["relationship"] in ["reply", "quote"]:
                # next we look at if frame is identified in target
                if dyad["target_frames"][frame] == 1:
                    target_positive[dyad["relationship"]] += 1

                    if dyad["source_frames"][frame] == 1:
                        source_and_target[dyad["relationship"]] += 1

        return_probs = {}
        for kind in ["reply", "quote"]:
            # if we never observe a frame in the target we want nan
            if target_positive[kind] > 0:
                return_probs[kind] = source_and_target[kind] / target_positive[kind]
            else:
                return_probs[kind] = np.nan
        
        return return_probs

def test_frame_conditional_prob():
    data = [{"source_frames": {"a": 1}, "target_frames": {"a": 0}, "relationship": "quote"},
            {"source_frames": {"a": 1}, "target_frames": {"a": 1.0}, "relationship": "reply"},
            {"source_frames": {"a": 0.0}, "target_frames": {"a": 1}, "relationship": "reply"}]
    
    print(frame_conditional_prob("a", data, by_kind=True)["reply"] == 0.5)
    print(frame_conditional_prob("a", data, by_kind=True)["quote"])
    assert frame_conditional_prob("a", data, by_kind=True)["reply"] == 0.5

test_frame_conditional_prob()

# %% [markdown]
#
# Now let's try running this and see if we see more agreement with replies
#

# %%
probs = {kind: {} for kind in ["quote", "reply"]}
for frame in all_frames:
    prob_kinds = frame_conditional_prob(frame, dyads, by_kind=True)
    probs["reply"][frame] = prob_kinds["reply"]
    probs["quote"][frame] = prob_kinds["quote"]

print(json.dumps(probs, indent=2))

# %% [markdown]
#
# ## Visualize the probability differences
#
# Easier to see at a glance if we just make a bar plor or something

# %%
import seaborn as sns

conditional_probs = pd.DataFrame(probs).reset_index().rename({"index": "frame"}, axis="columns")

conditional_probs["order"] = conditional_probs[["quote", "reply"]].max(axis="columns")

conditional_probs = (conditional_probs.sort_values(by="order", ascending=False)
                                      .drop("order", axis="columns"))

# lets also drop any pairs where there are no nonzero definate probabilities
conditional_probs = conditional_probs.fillna(0)[(conditional_probs[["quote", "reply"]] != 0).all(1)]


long_probs = pd.melt(conditional_probs, id_vars="frame",
                     value_vars=["quote", "reply"],
                     var_name="Relationship", value_name="prob")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
sns.barplot(long_probs, y="frame", x="prob",
            hue="Relationship",
            orient="horizontal")
plt.xlabel("Conditional Probability")
plt.tight_layout()
plt.savefig("plots/frame_conditional_probability.png")
# %% [markdown]

# Ok what about just in general how different are the sets of frames across
# dyads. Can quantify this with hamming distance. I'd like to compare this
# to distance between users in frame space but we'll see.

# %%
# we dont really need this import for hamming but it will be nice for euclidian
from scipy.spatial import distance as dist

def frame_difference(dyad: Dict, normalize=True) -> float:
    # pull out frame arrays
    sf = []
    tf = []
    for key in dyad["source_frames"].keys():
        sf.append(dyad["source_frames"][key])
        tf.append(dyad["target_frames"][key])

    hamming = dist.hamming(sf, tf)

    return hamming



# %% [markdown]
#
# Ok let's use this bad boy. We'll do something similar to what we've done
# above. Separate quotes and retweets and look at, I guess distributions this
# time. See if they look different. We're still dancing around the interesting
# stuff but I've got to do something.

# %%
dists = []
labels = []
for dyad in dyads:
    if dyad["relationship"] != "retweet":
        dists.append(frame_difference(dyad, normalize=False))
        labels.append(dyad["relationship"])


# %%
dist_df = pd.DataFrame({"relationship": labels, "distance": dists})

plt.hist(dist_df[dist_df["relationship"] == "quote"]["distance"], density=True, alpha=0.3)
plt.hist(dist_df[dist_df["relationship"] == "reply"]["distance"], density=True, alpha=0.3)
plt.show()

# %% [markdown]
#
# Looks pretty indistinguishable to me. What is more interesting is probably
# the difference across dyads compared to the user distances in frame space???
#

# %%
avg_frames = pd.read_csv("data/user_info/user_avg_frames.tsv", sep="\t")

def parse_screen_names(dyad: Dict) -> (str, str):
    screen_names = []
    for end in ["source_full", "target_full"]:
        if "user" in dyad[end]:
            screen_names.append(dyad[end]["user"]["screen_name"])
        elif "screen_name" in dyad[end]:
            screen_names.append(dyad[end]["screen_name"])
        else:
            raise KeyError("Unknown JSON structure!")
        
    return (screen_names[0], screen_names[1])


labels = []
hammings = []
dists = []

for dyad in dyads:
    if dyad["relationship"] != "retweet":
        
        source_name, target_name = parse_screen_names(dyad)

        source_avg = avg_frames[avg_frames["screen_name"] == source_name].drop("screen_name", axis="columns").values
        target_avg = avg_frames[avg_frames["screen_name"] == target_name].drop("screen_name", axis="columns").values

        if source_avg.shape[0] == 0 or target_avg.shape[0] == 0:
            continue
        else:
            dists.append(dist.euclidean(source_avg[0], target_avg[0]))
            hammings.append(frame_difference(dyad))
            labels.append(dyad["relationship"])

dist_df = pd.DataFrame({"kind": labels, "avg_dist": dists, "hamming": hammings})

# %%
import scipy.stats as sst
sns.scatterplot(dist_df, x="avg_dist", y="hamming", hue="kind", alpha=0.6)
plt.xlabel("User Distance")
plt.ylabel("Dyad Framing Difference")
plt.savefig("plots/distance_to_distance.png")
plt.show()

quote_df = dist_df[dist_df["kind"] == "quote"]
reply_df = dist_df[dist_df["kind"] == "reply"]

quote_r = sst.spearmanr(quote_df["avg_dist"],
              quote_df["hamming"])

reply_r = sst.spearmanr(reply_df["avg_dist"], reply_df["hamming"])

# %%
