# %%
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

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

# %%
import plotly.express as px
frame = "Threat: Jobs"
fig = px.line(frame_ts, x="time_stamp", y=frame, title=frame,
              labels={
                  "time_stamp": "Date",
                  frame: "Proportion of daily tweets cuing frame"
              })

fig.show()
# %% [markdown]
# ## Approach to finding parameters for `find_peaks()`
#
# We basically have 5 basic parameters to play with in this function:
#
# * `width` - the number of timesteps that must be peaky to count as a peak
# * `height` - an absolute threshold that peaks are above (probably not useful)
# * `threshold` - a threshold above a points neighbors that defines a peak. closer maybe?
# * `distance` - number of timesteps between allowed detected peaks
# * `prominence` - its something like the amount of descending one must do 
# from a peak before one can begin climbing to a higher peak. apparently very useful
#
# So what we're going to do is visualize some parameter sweeps of these
# variables and pick out places that look good on a particular time series.
# Because the number of cues vary a great deal between frames we will normalize
# each time series individually so that all of the parameters that deal with
# height are on a similar scale. Otherwise we would have to refit something like
# prominence for each frame.
#
# %%
from scipy.signal import find_peaks
frame = "Threat: Jobs"


def plot_find_peaks_parameter_sweep(sweep_time_series_list,
                                    sweep_parameters,
                                    sweep_variable_name,
                                    time_axis,
                                    num_tweets):

    fig, ax = plt.subplots(nrows=4, figsize=(7, 11))

    for i, (peaks, par) in enumerate(zip(sweep_time_series_list,
                                         sweep_parameters)):
        ax[i].plot(time_axis, num_tweets)
        ax[i].scatter(time_axis[peaks[0]], num_tweets[peaks[0]],
                   marker="x", c="k", s=50)
        ax[i].set_title(f"`{sweep_variable_name}` = {par}")
        ax[i].set_ylabel("Num. Tweets [normalized]")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## `width`
#
# I dont think we care about this variable. I dont think there is any reason to
# believe that a frame must be elevated for multiple days to consider it a peak
# We'll still play with it and see what we see.
#
# %%
frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
frame_sums = frame_ts[frame]
time_stamps = frame_ts["time_stamp"]

param_sweep = []
params = [1, 2, 3, 4]
for i, par in enumerate(params):
    param_sweep.append(find_peaks(frame_sums_normed, width=par))

plot_find_peaks_parameter_sweep(param_sweep, params, "width", time_stamps, frame_sums)
# %% [markdown]
# So a width of two kind of looks the best to me because it gets rid of a bit
# of the background noise when the frame is clearly not peaking. However, as
# stated above I'm not sure this is how we want to do this filtering.
#
# ## `height`
#
# Again I think this is not the way. Absolute height threshold seems like a
# really naive approach when we can do almost anything to account for noise
# floors. Still, we can try.
# %%
frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
frame_sums = frame_ts[frame]
time_stamps = frame_ts["time_stamp"]

param_sweep = []
params = [0.1, 0.2, 0.3, 0.4]
for i, par in enumerate(params):
    param_sweep.append(find_peaks(frame_sums_normed, height=par))

plot_find_peaks_parameter_sweep(param_sweep, params, "height", time_stamps, frame_sums)
# %% [markdown]
# I suppose that this could be used along with other filters to straight up
# exclude a bunch of small stuff. I'm not sure we have to mess with it though.
#
# ## `threshold`
#
# I suspect that this will be straight up better than `height` because it can
# account for some of the background at least locally.
# %%
frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
frame_sums = frame_ts[frame]
time_stamps = frame_ts["time_stamp"]

param_sweep = []
params = [0.01, 0.02, 0.05, 0.1]
for i, par in enumerate(params):
    param_sweep.append(find_peaks(frame_sums_normed, threshold=par))

plot_find_peaks_parameter_sweep(param_sweep, params, "threshold", time_stamps, frame_sums)
# %%
# Ok so this seems actually not useful to me! Even at very small values it
# excludes what to me is the first obvious peak in June 2018. Weird filter, I
# don't like it.
#
# ## `distance`
#
# I think this could be very useful but we need to be careful because this
# could aggressiuvely and directly limit the number of peaks we find.

# %%
frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
frame_sums = frame_ts[frame]
time_stamps = frame_ts["time_stamp"]

param_sweep = []
params = [1, 7, 14, 30]
for i, par in enumerate(params):
    param_sweep.append(find_peaks(frame_sums_normed, distance=par))

plot_find_peaks_parameter_sweep(param_sweep, params, "distance", time_stamps, frame_sums)
# %%
# I kind of think there may be some justification for this kind of filtering.
# like some kind of limit for how quickly the discourse can change on a big
# platform like twitter. However, I think it would be best to avoid this
# parameter if possible.
#
# ## `prominence`
#
# This is supossed to be the magic one according to this: https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy/
# let's give it a try. I have no idea what to expect when varying this.

# %%
frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
frame_sums = frame_ts[frame]
time_stamps = frame_ts["time_stamp"]

param_sweep = []
params = [0.05, 0.15, 0.2, 0.3]
for i, par in enumerate(params):
    param_sweep.append(find_peaks(frame_sums_normed, prominence=par))

plot_find_peaks_parameter_sweep(param_sweep, params, "prominence", time_stamps, frame_sums)

# %% [markdown]
#
# Ok I do think `prominence`` is the one. I think maybe we should combine it
# with a height filter and use like prominence around 0.15. Well no I guess
# prominence has has a built in hight filter. One cannot descend 0.1 if the
# candidate peak is less than 0.1. Let's do a smaller sweep from say, 0.1 to
# 0.2 to see how it looks.
# %%
frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
frame_sums = frame_ts[frame]
time_stamps = frame_ts["time_stamp"]

param_sweep = []
params = [0.1, 0.125, 0.15, 0.175]
for i, par in enumerate(params):
    param_sweep.append(find_peaks(frame_sums_normed, prominence=par))

plot_find_peaks_parameter_sweep(param_sweep, params, "prominence", time_stamps, frame_sums)

# %%
# Ok I think we should go with the prominence of 0.125 and see how it
# looks on another time series. That value doesn't exclude anything that to me
# looks like an obvious peak with my feeble human brain. Real quick before we
# do that let's just take a quick look as how it looks if we combine a lienient
# of prominence with some best values of the other params to see if it looks
# better. Just to be thorough. We'll use 0.5 prominence because adding an
# additional filter can only remove peaks. Basically we want to see if any of
# these look better or as good.
# %%
frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
frame_sums = frame_ts[frame]
time_stamps = frame_ts["time_stamp"]



fig, ax = plt.subplots(nrows=4, figsize=(7, 11))

# just prominence = 0.125
peaks = find_peaks(frame_sums_normed, prominence=0.125)
ax[0].plot(time_stamps, frame_sums)
ax[0].scatter(time_stamps[peaks[0]], frame_sums[peaks[0]],
            marker="x", c="k", s=50)
ax[0].set_title(f"`prominence` = 0.125")
ax[0].set_ylabel("Num. Tweets")

peaks = find_peaks(frame_sums_normed, prominence=0.05, width=2)
ax[1].plot(time_stamps, frame_sums)
ax[1].scatter(time_stamps[peaks[0]], frame_sums[peaks[0]],
            marker="x", c="k", s=50)
ax[1].set_title(f"`prominence` = 0.05, width=2")
ax[1].set_ylabel("Num. Tweets")


peaks = find_peaks(frame_sums_normed, prominence=0.05, threshold=0.02)
ax[2].plot(time_stamps, frame_sums)
ax[2].scatter(time_stamps[peaks[0]], frame_sums[peaks[0]],
            marker="x", c="k", s=50)
ax[2].set_title(f"`prominence` = 0.05, threshold=0.02")
ax[2].set_ylabel("Num. Tweets")


peaks = find_peaks(frame_sums_normed, prominence=0.05, distance=3)
ax[3].plot(time_stamps, frame_sums)
ax[3].scatter(time_stamps[peaks[0]], frame_sums[peaks[0]],
            marker="x", c="k", s=50)
ax[3].set_title(f"`prominence` = 0.05, distance=3")
ax[3].set_ylabel("Num. Tweets")

plt.tight_layout()
plt.show()

# %% [markdown]
# Ok all of the combinations either miss obvious peaks or find bad ones. We'll
# just use prominence = 0.15. Let's look at a few different frames now just
# to see how they look

# %%
frames = ["Crime and Punishment",
          "Victim: Humanitarian",
          "Political Factors and Implications"]

fig, ax = plt.subplots(nrows=3, figsize=(7, 11))

for i, frame in enumerate(frames):
    frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
    frame_sums = frame_ts[frame]
    time_stamps = frame_ts["time_stamp"]

    peaks = find_peaks(frame_sums_normed, prominence=0.15)
    ax[i].plot(time_stamps, frame_sums)
    ax[i].scatter(time_stamps[peaks[0]], frame_sums[peaks[0]],
            marker="x", c="k", s=50)
    ax[i].set_title(frame)
    ax[i].set_ylabel("Num. Tweets")

plt.tight_layout()
plt.show()
# %% [markdown] 
# Ok this looks good to me. Now we need to export all of these
# dates into a table of some kind so we can think about catalogging events.
# We'll make a csv that contains the columns `date`, `frame`, `num_tweets` and
# save it as a tsv for manipulation in excel I suppose. We'll probably also
# want to consolidate by date but we can figure that out next. First step is
# just making sure we save this work.
#
# We may have a few more peaks than we want but better to have too many than
# too few i think.
# 
# Of course we'll also want to get it into a script and snakemake eventually.
# %%
prom_filter = 0.2
rows = []
for frame in all_frame_list:
    frame_sums_normed = frame_ts[frame] / frame_ts[frame].max()
    frame_sums = frame_ts[frame]
    time_stamps = frame_ts["time_stamp"]

    for peak_idx in find_peaks(frame_sums_normed, prominence=prom_filter)[0]:
        row_date = time_stamps[peak_idx].date()
        row = {"date": time_stamps[peak_idx].date(),
               "frame": frame,
               "num_tweets": frame_sums[peak_idx]}
        rows.append(row)

events_long = pd.DataFrame(rows)
events_long.to_csv("data/events/found_peaks.tsv", sep="\t", index=False)
# %% [markdown]
#
# Now we can consolidate by date and see how many we have. if its a shit ton
# we will tighten up our prominence filter a bit. We need to have a feasible
# number of events to work with.
# %%
consolidated_dates = events_long[["date"]].drop_duplicates()
consolidated_dates.to_csv("data/events/consolidated_dates.csv", index=False)
# %% [markdown]
#
# Finally, let's just look at our events timeline and how frequently they
# appear.
# %%
fig, ax = plt.subplots()
ax.plot(time_stamps, [0]*time_stamps.shape[0])
ax.vlines(consolidated_dates, 0, 1, color="k", linewidth=0.5)
plt.show()
# %%
