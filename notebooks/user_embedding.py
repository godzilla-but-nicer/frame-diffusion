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


with open("workflow/sample_paths.json", "r") as pf:
    paths = json.loads(pf.read())
print("config and paths loaded")


all_frames = pd.read_csv(paths["all_frames"], sep="\t")

# %%
# subsample the public
public_sample_fraction = 1
required_observations = 10
sampled_public = all_frames[all_frames["group"] == "public"].sample(frac=public_sample_fraction)
filtered_public = sampled_public.groupby("screen_name").filter(lambda x: len(x) > required_observations)

# drop the public from the all frames df and add back the sample
sampled_frames = all_frames[all_frames["group"] != "public"]
sampled_frames = pd.concat((sampled_frames, filtered_public)).dropna(axis="columns", how="any")

# %%
complete_frame_cols = ['Capacity and Resources',
       'Crime and Punishment', 'Cultural Identity', 'Economic',
       'External Regulation and Reputation', 'Fairness and Equality',
       'Health and Safety', 'Legality, Constitutionality, Jurisdiction',
       'Morality and Ethics', 'Policy Prescription and Evaluation',
       'Political Factors and Implications', 'Public Sentiment',
       'Quality of Life', 'Security and Defense', 'Hero: Cultural Diversity', 'Hero: Integration', 'Hero: Worker',
       'Threat: Fiscal', 'Threat: Jobs', 'Threat: National Cohesion',
       'Threat: Public Order', 'Victim: Discrimination',
       'Victim: Global Economy', 'Victim: Humanitarian', 'Victim: War']

thin_frames = sampled_frames.drop(["id_str", "time_stamp"], axis="columns")
frame_means = thin_frames.groupby(["screen_name", "group"]).mean().reset_index().dropna()

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(frame_means[complete_frame_cols].values)

non_congress = X_pca[frame_means["group"] != "congress"]
congress = X_pca[frame_means["group"] == "congress"]


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.05)

pca.explained_variance_ratio_[0]

ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)")
ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)")
fig.savefig("plots/structure/frame-umap.png")


# %%
congress_info = pd.read_csv("data/user_info/congress_info.tsv", sep="\t")[["screen_name", "party"]]

info_frames = pd.merge(frame_means, congress_info, on="screen_name", how="left")
info_frames = info_frames[info_frames["group"] == "congress"]

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

# %%
import warnings
warnings.filterwarnings("error")

import powerlaw

user_spreads = []
normed_user_spreads = []
users = []
groups = []
for i, user in enumerate(tqdm(all_frames["screen_name"].unique())):
    user_frames = all_frames[all_frames["screen_name"] == user]
    if user_frames.shape[0] < 30:
        continue

    try:
        user_spreads.append(frame_spread(user_frames))
        normed_user_spreads.append(frame_spread(user_frames) / user_frames.shape[0])
        users.append(user)
        groups.append(user_frames["group"].unique()[0])
    
    except RuntimeWarning as w:
        print(w)
        print(user_frames)
        break

    except ValueError as e:
        print(e)
        print(user_frames)
        break

spread_array = np.array(user_spreads)
normed_array = np.array(normed_user_spreads)
user_array = np.array(users)
group_array = np.array(groups)
# %%
vals, counts = np.unique(spread_array, return_counts=True)
cdf = np.cumsum(counts) / np.sum(counts)

plt.set_title("Overall Distribution")
plt.scatter(vals, 1-cdf)
plt.xscale("log")
plt.yscale("log")
plt.show()

# %%
congress_spreads = normed_array[group_array == "congress"]
journalist_spreads = normed_array[group_array == "journalists"]
trump_spread = normed_array[group_array == "trump"]

c_vals, c_counts = np.unique(congress_spreads, return_counts=True)
c_cdf = np.cumsum(c_counts) / np.sum(c_counts)

j_vals, j_counts = np.unique(journalist_spreads, return_counts=True)
j_cdf = np.cumsum(j_counts) / np.sum(j_counts)

plt.scatter(c_vals, 1 - c_cdf, label = "congress")
plt.scatter(j_vals, 1 - j_cdf, label = "journalists")
plt.axvline(trump_spread, label="trump", c="green")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frame Spread")
plt.ylabel("P(Frame Spread)")
plt.legend()
plt.show()
# %%
user_array
# %% [markdown]

