# %%
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
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

with open("workflow/sample_paths.json", "r") as pf:
    paths = json.loads(pf.read())

user_id_map = pd.read_csv("data/down_sample/user_id_map.tsv", sep="\t",
                          dtype={"screen_name": str, "user_id": str})

mentions = pd.read_csv("data/down_sample/edge_lists/in_sample_mentions.tsv", sep="\t",
                       dtype={"uid1": str, "uid2": str,
                              "1to2freq": int, "2to1freq": int})
# %%
# the first time we run this let's just save all the frames
# f = fs.load_all_frames(paths, config)
# f.to_csv("data/down_sample/binary_frames/all_frames.tsv", sep="\t", index=False)

# then we can load them instead of building the df

f = pd.read_csv("data/down_sample/binary_frames/all_frames.tsv", sep="\t")
# %%
filtered_tweets = fs.filter_users_by_activity(f, 10)

# %% [markdown]
#
# ## Self imposed influence
#
# We're going to calculate the odds ratio: P(cue|treated) / P(cue|untreated) for each
# frame where treated means a user cued the frame on both day t and t + 1
# and untreated means the user cued the frame on t + 1 but not t.
#
# To calculate these values we need to calculate, for example:
#
# $$
# P(cue|treated) = \frac{P(cue \cap treated)}{P(treated)}
# $$
#
# We get the numerator by counting the number of day pairs with the frame cued
# in both day t and t + 1 and we get the denominator by counting the number
# of pairs with the frame cued in time t. We don't actually have to normalize
# because they are both normalized by the same value (total number of pairs)

# %%
# this part takes a couple of minutes
all_frame_pairs = []
for user in tqdm(filtered_tweets["screen_name"].unique()):
    user_time_series_df = ts.construct_frame_time_series(filtered_tweets,
                                                         user,
                                                         "1D",
                                                         config)
    user_frame_pairs = ci.construct_frame_pairs(user_time_series_df)
    all_frame_pairs.extend(user_frame_pairs)

# %%
def get_treated_untreated_contingency_table(frame_pairs: List[Dict], frame: str):
    
    # the table will look like this
    # __________| cued   | not cued |
    # treated   | n[0,0] | n[0,1]   |
    # untreated | n[1,0] | n[1,1]   |

    # variables to hold our counts
    n = np.zeros((2,2))

    for pair in frame_pairs:

        if (pair["t"][frame] > 0) & (pair["t+1"][frame] > 0):
            n[0, 0] += 1
        
        elif (pair["t"][frame] > 0) & (pair["t+1"][frame] == 0):
            n[0, 1] += 1

        elif (pair["t"][frame] == 0) & (pair["t+1"][frame] > 0):
            n[1, 0] += 1

        elif (pair["t"][frame] == 0) & (pair["t+1"][frame] == 0):
            n[1, 1] += 1
    
    return n
# %%
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]
# %%
odds_ratios = []
odds_ratios_se = []
p_values = []
for frame in tqdm(all_frame_list):
    ct = get_treated_untreated_contingency_table(all_frame_pairs, frame)
    table = sm.stats.Table2x2(ct)
    odds_ratios.append(table.log_oddsratio)
    odds_ratios_se.append(table.log_oddsratio_se)
    p_values.append(table.log_oddsratio_pvalue())
    
frame_odds_df = pd.DataFrame({"frame": all_frame_list, 
                              "log_odds_ratio": odds_ratios,
                              "log_odds_ratio_se": odds_ratios_se,
                              "p_value": p_values})

print("User self-influence")
print(frame_odds_df)
frame_odds_df.to_csv("self_influence_odds.tsv", index=False, sep="\t")
# %%
# now we will repeat for the second experiment
# this part takes a couple of minutes
influencer_frame_pairs = []
for i, user in tqdm(enumerate(filtered_tweets["screen_name"].unique())):
    user_time_series_df = ts.construct_frame_time_series(filtered_tweets,
                                                         user,
                                                         "1D",
                                                         config)
    influencer_time_series_dfs = ts.get_influencer_time_series(user,
                                                               filtered_tweets,
                                                               config,
                                                               mentions,
                                                               user_id_map)
    user_frame_pairs = ci.construct_influencer_frame_pairs(user_time_series_df,
                                                           influencer_time_series_dfs,
                                                           1, "days")
    
    influencer_frame_pairs.extend(user_frame_pairs)
    
    if i > 300:
        break
    

# %%
odds_ratios = []
odds_ratios_se = []
p_values = []

for frame in tqdm(all_frame_list):
    ct = get_treated_untreated_contingency_table(influencer_frame_pairs, frame)
    table = sm.stats.Table2x2(ct)
    odds_ratios.append(table.log_oddsratio)
    odds_ratios_se.append(table.log_oddsratio_se)
    p_values.append(table.log_oddsratio_pvalue())

influencer_odds_df = pd.DataFrame({"frame": all_frame_list, 
                                   "log_odds_ratio": odds_ratios,
                                   "log_odds_ratio_se": odds_ratios_se,
                                   "p_value": p_values})

print("mention network influence")
influencer_odds_df
influencer_odds_df.to_csv("alter_influenced_odds.tsv", index=False, sep="\t")
# %%
frame_odds_df = pd.read_csv("self_influence_odds.tsv", sep="\t")
influencer_odds_df = pd.read_csv("alter_influenced_odds.tsv", sep="\t")
# %%
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import seaborn as sns
# %%
# first the self-influence
frame_odds_df["p_value_corrected"] = frame_odds_df["p_value"] * frame_odds_df.shape[0]
frame_odds_df["signif"] = frame_odds_df["p_value_corrected"] < 1e-5

frame_odds_sorted = frame_odds_df.sort_values("log_odds_ratio", ascending=False)


sns.catplot(frame_odds_sorted, y="frame", x="log_odds_ratio",
            kind="bar", color="C0",
            height=5, aspect=1.2)

# add asterisks for significance
y_coords = np.arange(frame_odds_sorted.shape[0])
x_coords = list(frame_odds_sorted["log_odds_ratio"] + 0.1)
symbol = ["*" if sig == True else " " for sig in frame_odds_sorted["signif"]]

for i in range(frame_odds_sorted.shape[0]):
    plt.text(x_coords[i], y_coords[i]+0.35, s=symbol[i])

plt.xlabel("Log Odds Ratio")
plt.ylabel("")
plt.title("Self-influence")

plt.tight_layout()

plt.savefig("self_influence_odds_ratio.pdf")
plt.savefig("self_influence_odds_ratio.png")
plt.show()
# %%
# now the alter influence
influencer_odds_df["p_value_corrected"] = influencer_odds_df["p_value"] * influencer_odds_df.shape[0]
influencer_odds_df["signif"] = influencer_odds_df["p_value_corrected"] < 1e-5

influencer_odds_sorted = influencer_odds_df.sort_values("log_odds_ratio", ascending=False)


sns.catplot(influencer_odds_sorted, y="frame", x="log_odds_ratio",
            kind="bar", color="C0",
            height=5, aspect=1.2)

# add asterisks for significance
y_coords = np.arange(influencer_odds_sorted.shape[0])
x_coords = list(influencer_odds_sorted["log_odds_ratio"] + 0.1)
symbol = ["*" if sig == True else " " for sig in influencer_odds_sorted["signif"]]

for i in range(influencer_odds_sorted.shape[0]):
    plt.text(x_coords[i], y_coords[i]+0.35, s=symbol[i])

plt.xlabel("Log Odds Ratio")
plt.ylabel("")
plt.title("Alter-influence")
plt.savefig("alter_influence_odds_ratio.pdf")
plt.savefig("alter_influence_odds_ratio.png")
plt.show()

# %%
