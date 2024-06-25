# %%
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import timedelta

import frame_stats as fs


# we dont want to be working in notebooks/ for pathing reasons
import os
if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")
print(f"Working in {os.getcwd()}")

# config has frame names
print("loading config and paths")
with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# paths to data files
with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())
print("config and paths loaded")

# load all of the frames and tweet time stamps etc.
print("loading tweets")
f = pd.read_csv(paths["all_frames"], sep="\t")
print("tweets loaded")

# list of all frame names
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]

# %%
frame_times = f[["time_stamp"] + all_frame_list]


# ensure that the time stamp column is being stored correctly
frame_times["time_stamp"] = pd.to_datetime(frame_times["time_stamp"], utc=True)

frame_ts = (frame_times.groupby(pd.Grouper(key="time_stamp",
                                                freq="1D"))
                    .sum()
                    .reset_index())

frame_ts["Generic"] = np.sum(frame_ts[config["frames"]["generic"]], axis=1)
frame_ts["Specific"] = np.sum(frame_ts[config["frames"]["generic"]], axis=1)
frame_ts["Non-Narrative"] = frame_ts["Generic"] + frame_ts["Specific"]

frame_ts["time_stamp"] = frame_ts["time_stamp"].dt.date
# %%
events = pd.read_csv("data/events/consolidated_dates_first.csv").dropna(subset="resolved")
events["date"] = pd.to_datetime(events["date"])

# %%
frame = "Non-Narrative"
plt.figure(figsize=(12, 4))
for _, e in events.iterrows():
    if e["resolved"] == True:
        plt.axvline(e["date"], c="k", alpha=0.4, lw=1)
    else:
        plt.axvline(e["date"], c="gray", alpha=0.4, lw=1)

plt.plot(frame_ts["time_stamp"], frame_ts[frame])
plt.tight_layout()
plt.xlabel("Num. Frames")
plt.savefig("plots/events/time_series_with_events.png")
plt.savefig("plots/events/time_series_with_events.pdf")

# %%
# get the responses to the events
event_response_rows = []

time_window = 5  # days
for _, e in events.iterrows():

    new_row = {}

    new_row["date"] = e["date"]
    new_row["category"] = e["category"]
    new_row["resolved"] = e["resolved"]

    response_window = frame_ts[(frame_ts["time_stamp"] >= e["date"]) & 
                               (frame_ts["time_stamp"] <= e["date"] + timedelta(days=time_window))]
    frames = response_window[all_frame_list].sum(axis=0)

    new_row.update(frames)
    event_response_rows.append(new_row)

event_response_df = pd.DataFrame(event_response_rows)
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# let's keep only rows i think are fully resolved and that have category labels
event_response_plot = event_response_df[event_response_df["resolved"]].drop(["resolved"], axis="columns")
event_response_plot = event_response_plot.dropna(subset="category")
event_response_plot = event_response_plot.drop(["Episodic", "Thematic"], axis="columns")

pca = PCA(n_components=2)
sts = StandardScaler()

X_all = event_response_plot.drop(["date", "category"], axis="columns").values
X_all_std = sts.fit_transform(X_all)
X_pca = pca.fit_transform(X_all_std)

plot_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1]})
plot_df["response_size"] = (plot_df["PC1"]**2 + plot_df["PC2"]**2)**(1/2)
plot_df["date"] = event_response_plot["date"]
plot_df["category"] = event_response_plot["category"]

sns.scatterplot(plot_df, x="PC1", y="PC2", hue="category")
plt.savefig("plots/events/response_pca.png")
plt.savefig("plots/events/response_pca.pdf")
# %%
event_activity = pd.read_csv("data/events/found_peaks.tsv", sep="\t")
event_activity["date"] = pd.to_datetime(event_activity["date"])
event_activity_wide = pd.pivot(event_activity,
                               index="date",
                               columns="frame",
                               values="num_tweets").fillna(0)

events = events[events["resolved"]]

described_activity = pd.merge(event_activity_wide, events[["date"]], on="date")
described_activity

pca = PCA(n_components=2)
sts = StandardScaler()

X = described_activity.drop(["date", "Episodic", "Thematic"], axis="columns").values
X_std = sts.fit_transform(X)
X_pca = pca.fit_transform(X_std)

explained = pca.explained_variance_ratio_

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1])
ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f} %)")
ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f} %)")
plt.show()
# %%
