# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa import ar_model
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import stattools
from scipy.stats import lognorm, poisson, expon
import data_selector as ds

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import permutations


if os.getcwd().split("/")[-1] == "scripts" or os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("../")

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())


with open("workflow/paths.json", "r") as cf:
    paths = json.loads(cf.read())


# normalize the windowed sums by the total number of frames for the group in
# the given time window. each window within a group will then represent a
# probability distribution of frames for the group
normalize = True
# %%

public = ds.load_public_frames()
public = public[~public["ideology"].isna()]
congress = ds.load_congress_frames()
journalist = ds.load_journalist_frames()
trump = ds.load_trump_frames()

# %%
# now we want to break it out into partisan groups
frames = {k: {} for k in ["public", "congress", "journalist"]}

frames["public"]["left"] = public[public["ideology"] < 0]
frames["public"]["right"] = public[public["ideology"] > 0]
frames["congress"]["left"] = congress[congress["Affiliation"] == "Democrat"]
frames["congress"]["right"] = congress[congress["Affiliation"] == "Republican"]
frames["journalist"]["left"] = journalist[journalist["bias_rating"] < 0]
frames["journalist"]["right"] = journalist[journalist["bias_rating"] > 0]
frames["trump"] = trump

# %%
measure_frequency = "8H"
frame_labels = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]
good_frames = [frame for frame in frame_labels if frame not in config["frames"]["low_f1"]]

period_sums = {k: {} for k in ["public", "congress", "journalist"]}

for group in ["public", "congress", "journalist"]:    
    for politics in ["left", "right"]:

        frames_and_time = frames[group][politics][good_frames + ["time_stamp"]].copy()
        frames_and_time["time_stamp"] = pd.to_datetime(frames_and_time["time_stamp"], utc=True)
        period_sums[group][politics] = (frames_and_time.groupby(pd.Grouper(key="time_stamp",
                                                                freq=measure_frequency))
                                        .sum()
                                        .reset_index())
        

        if normalize:
            # pull out time labels
            time_stamps = period_sums[group][politics]["time_stamp"]

            # normalize rows
            frames_only = period_sums[group][politics].drop("time_stamp", axis="columns")

            # avoid divide by zero
            row_sums = frames_only.sum(axis="columns")
            row_sums[row_sums == 0] = 1

            # normalize
            normed_frames = frames_only.div(row_sums, axis="rows")

            # glue back together
            period_sums[group][politics] = pd.concat((time_stamps, normed_frames),
                                            axis="columns")
            
            # save the data
            period_sums[group][politics].to_csv(f"data/time_series_intermediaries/group_sums/period_sums_{group}_{politics}_normalized.csv")

        else:
            period_sums[group][politics].to_csv(f"data/time_series_intermediaries/group_sums/period_sums_{group}_{politics}.csv")

# now for trump
frames_and_time = frames["trump"][good_frames + ["time_stamp"]].copy()
frames_and_time["time_stamp"] = pd.to_datetime(frames_and_time["time_stamp"], utc=True)
period_sums["trump"] = (frames_and_time.groupby(pd.Grouper(key="time_stamp",
                                                        freq=measure_frequency))
                                .sum()
                                .reset_index())

group = "trump"
if normalize:
    # pull out time labels
    time_stamps = period_sums[group]["time_stamp"]
    # normalize rows
    frames_only = period_sums[group].drop("time_stamp", axis="columns")
    
    # avoid divide by zero
    row_sums = frames_only.sum(axis="columns")
    row_sums[row_sums == 0] = 1
    # normalize
    normed_frames = frames_only.div(row_sums, axis="rows")
    # glue back together
    period_sums[group] = pd.concat((time_stamps, normed_frames),
                                    axis="columns")

    # save data
    print("here")
    period_sums[group].to_csv(f"data/time_series_intermediaries/group_sums/period_sums_trump_normalized.csv")

else:
    print("there")
    period_sums[group].to_csv(f"data/time_series_intermediaries/group_sums/period_sums_trump.csv")


# %% [markdown]
# ## Fitting AR models
#
# We're going to skip the fun squishy stuff and just optimize the fits

# %%
residuals = {k: {} for k in ["public", "congress", "journalist", "trump"]}
all_periods = pd.DataFrame(period_sums["congress"]["left"]["time_stamp"])

ts_lags = []

for frame in good_frames:
    for group in ["public", "congress", "journalist"]:
        for politics in ["left", "right"]:

            if politics not in residuals[group]:
                residuals[group][politics] = {}
            
            time_series = period_sums[group][politics][["time_stamp", frame]]

            time_series_fixed = pd.merge(all_periods, time_series,
                                         on="time_stamp", how="left").fillna(0)
            
            
            order = ar_model.ar_select_order(time_series_fixed[frame],
                                             maxlag=21,
                                             seasonal=True,
                                             period=3)
            
            if order.ar_lags:

                selected_lags = order.ar_lags
                ar_fit = ar_model.AutoReg(time_series_fixed[frame],
                                          lags=selected_lags,
                                          seasonal=True,
                                          period=3).fit()
                
                residuals[group][politics][frame] = ar_fit.resid

            else:

                residuals[group][politics][frame] = time_series_fixed[frame]
            
            # update the the lags dataframe
            ts_lags.append({"frame": f"{group}_{politics}_{frame}", 
                            "ts_length": residuals[group][politics][frame].shape[0],
                            "fit_lag": max(ar_fit.ar_lags)})


# now for trump
for frame in good_frames:
    time_series = period_sums["trump"][["time_stamp", frame]]

    time_series_fixed = pd.merge(all_periods, time_series,
                                    on="time_stamp", how="left").fillna(0)
    
    order = ar_model.ar_select_order(time_series_fixed[frame],
                                        maxlag=21,
                                        seasonal=True,
                                        period=3)
    
    if order.ar_lags:

        selected_lags = order.ar_lags
        ar_fit = ar_model.AutoReg(time_series_fixed[frame],
                                    lags=selected_lags,
                                    seasonal=True,
                                    period=3).fit()

        residuals["trump"][frame] = ar_fit.resid

    else:

        residuals["trump"][frame] = time_series_fixed[frame]
    
    # update the the lags dataframe
    ts_lags.append({"frame": f"{group}_{politics}_{frame}", 
                    "ts_length": residuals[group][politics][frame].shape[0],
                    "fit_lag": max(ar_fit.ar_lags)})

# get the numbers we need for trimming the risiduals
residual_info_dataframe = pd.DataFrame(ts_lags)
max_lag = max(residual_info_dataframe["fit_lag"])
num_steps = all_periods.shape[0]
target_size = num_steps - max_lag

# now we need do trim off the raggedy beginnings of the residual timeseries
for group_key in residuals.keys():
    
    if group_key != "trump":
    
        for politics_key in residuals[group_key].keys():
            for frame_key in residuals[group_key][politics_key].keys():
                start_index = residuals[group_key][politics_key][frame_key].shape[0] - target_size
                residuals[group_key][politics_key][frame_key] = residuals[group_key][politics_key][frame_key][start_index:]

    else:
        for frame_key in residuals[group_key].keys():
            start_index = residuals[group_key][frame_key].shape[0] - target_size
            residuals[group_key][frame_key] = residuals[group_key][frame_key][start_index:]

keep_periods = all_periods[max_lag:]

events = pd.read_csv("data/events/consolidated_dates_first.csv")
events = events[events["resolved"] == True][["date"]]
events["time_stamp"] = pd.to_datetime(events["date"], utc=True)
events["event"] = 1
events = events.drop("date", axis="columns")

scaffolded_events = pd.merge(all_periods, events, on="time_stamp", how="left").fillna(0)[max_lag:]
# %%
from itertools import product
from typing import Tuple, Iterable
import warnings

groups = ["congress", "journalist", "public"]
politics = ["left", "right"]

group_politics = list(product(groups, politics))

categories = group_politics + ["trump"]

test_pairs = list(permutations(categories, 2))


def run_granger_causality(pair: Tuple,
                          data: dict,
                          events: Iterable,
                          frame: str,
                          shuffle_source: bool = False) -> dict:
    
    # we're going to return a dict we can use as a dataframe row
    result = {}
    result["frame"] = frame
    
    # pull out a dictionary for the data we are looking at
    if type(pair[0]) == tuple:
        source_data = data[pair[0][0]][pair[0][1]][frame].copy()
        result["source"] = pair[0][0] + "-" + pair[0][1]
    else:
        source_data = data[pair[0]][frame].copy()
        result["source"] = pair[0]
    
    if type(pair[1]) == tuple:
        target_data = data[pair[1][0]][pair[1][1]][frame].copy()
        result["target"] = pair[1][0] + "-" + pair[1][1]
    else:
        target_data = data[pair[1]][frame].copy()
        result["target"] = pair[1]

    if shuffle_source:
        source_data = np.random.permutation(source_data)
    
    gcdf = pd.DataFrame({"source": source_data, "target": target_data, "events": events}).fillna(0)
    

    if len(gcdf.columns[gcdf.nunique() <= 1]) > 0:
        return {}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        var_fit = VAR(gcdf).fit(maxlags=21, ic="aic")
    
    if var_fit.k_ar > 0:
        gc_without_events = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            gc_output = var_fit.test_causality("target", "source")
        gc_without_events["p_value"] = gc_output.pvalue
        gc_without_events["f_statistic"] = gc_output.test_statistic
        gc_without_events["events_causing"] = False
        no_event_result = result.copy()
        no_event_result.update(gc_without_events)

        gc_with_events = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            gc_output = var_fit.test_causality("target", ["source", "events"])
        gc_with_events["p_value"] = gc_output.pvalue
        gc_with_events["f_statistic"] = gc_output.test_statistic
        gc_with_events["events_causing"] = True
        result.update(gc_with_events)
    else:
        gc_without_events = {}
        gc_without_events["p_value"] = 1
        gc_without_events["f_statistic"] = 0
        gc_without_events["events_causing"] = False
        no_event_result = result.copy()
        no_event_result.update(gc_without_events)

        gc_with_events = {}
        gc_with_events["p_value"] = 1
        gc_with_events["f_statistic"] = 0
        gc_with_events["events_causing"] = True
        result.update(gc_with_events)

    return [no_event_result, result]


output_rows = []
for pair in tqdm(test_pairs):
    for frame in good_frames:
        output_rows.extend(run_granger_causality(pair, residuals, scaffolded_events["event"].values, frame))

granger_df = pd.DataFrame(output_rows).reset_index()
granger_df.to_csv(f"data/time_series_output/all_grangers_partisan_{normalize}.tsv", sep="\t")


alpha = 0.05

# multiple testing proceedure
def bonferroni_holm(data, alpha):
    sorted = data.sort_values("p_value")
    rejections = np.zeros(sorted.shape[0], dtype=bool)
    new_p = np.zeros(sorted.shape[0])

    for row_i in range(sorted.shape[0]):

        # functional alpha for each iteration
        abh = alpha / (sorted.shape[0] - row_i)
        new_p[row_i] = (sorted.iloc[row_i]["p_value"] * (sorted.shape[0] - row_i))

        if sorted.iloc[row_i]["p_value"] < abh:
            rejections[row_i] = True

    sorted["null_rejected"] = rejections
    sorted["p_corrected"] = new_p
    return sorted

gcdf_corrected = bonferroni_holm(granger_df, 0.05)
signif = gcdf_corrected[gcdf_corrected["null_rejected"] == True]
signif.to_csv(f"data/time_series_output/significant_complete_granger_partisan_{normalize}.tsv", sep="\t")

# %%
# bootstrap gc runs
output_rows = []
n_boots = 100
for pair in tqdm(test_pairs):
    for frame in good_frames:
        for _ in range(n_boots):
            output_rows.extend(run_granger_causality(pair, residuals, scaffolded_events["event"].values, frame, shuffle_source=True))

bootstrap_gcs = pd.DataFrame(output_rows)
bootstrap_gcs.to_csv("bootstrap_gc.tsv", sep="\t", index=False)
# %%
# This really should live else where but for now its here
bootstrap_gcs
# %%
observed_quantiles = []
for i, row in signif.iterrows():
    boots = bootstrap_gcs[(bootstrap_gcs["frame"] == row["frame"]) &
                          (bootstrap_gcs["source"] == row["source"]) &
                          (bootstrap_gcs["target"] == row["target"]) &
                          (bootstrap_gcs["events_causing"] == row["events_causing"])]
    
    observed_quantiles.append(np.mean(row["f_statistic"] < boots["f_statistic"]))

signif["bootstrap_p"] = observed_quantiles
boot_signif = signif[signif["bootstrap_p"] < 0.05]
signif.to_csv("data/time_series_output/significant_complete_granger_partisan_{normalized}_bootstrap.tsv")
# %%
print(np.sum(signif["events_causing"] == False))
print(np.sum(boot_signif["events_causing"] == False))
# %%
