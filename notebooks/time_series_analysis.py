# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa import ar_model
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


if os.getcwd().split("/")[-1] != "frame-diffusion":
    os.chdir("..")

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())


with open("workflow/paths.json", "r") as cf:
    paths = json.loads(cf.read())


# normalize the windowed sums by the total number of frames for the group in
# the given time window. each window within a group will then represent a
# probability distribution of frames for the group
normalize = True

# %%
frames = {}
frames["public"] = ds.load_public_frames()
frames["congress"] = ds.load_congress_frames()
frames["journalist"] = ds.load_journalist_frames()
frames["trump"] = ds.load_trump_frames()

# %% [markdown]
#
# # Time series analysis of Tweets
#
# In this notebook we will do all of the cool time series analysis that we will
# be doing on our aggregated groups of posters. We'll have to go through a few
# steps, first I want to see how the fluctuations are distributed. Then we'll
# look at autocorrelation to try and think about how to model these time
# series, then we'll conduct our granger causality tests after preprocessing
# however we need to.
#
# Before any of that we will get the number of tweets cuing each frame for each
# group within some time period. These are the time series we will be working
# with.
#

# %%
from functools import reduce

# eight-hour windows
measure_frequency = "8H"

# will hold keys for groups
period_sums = {}

for group in groups:
    if group == "public":

        all_frames = frames[group].drop(["id_str", "Unnamed: 0"], axis="columns")
        
    else:
        # let's get all of our frames in one place
        frame_type_dfs = []
        for frame_type in frame_types:
            df = frames[group][frame_type]
            df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True)
            frame_type_dfs.append(df)
        
        # fancy triple merge "one-liner"
        all_frames = (reduce(lambda l, r: pd.merge(l, r, on="time_stamp"),
                             frame_type_dfs).fillna(0))
        
    # get the frame sums in each time window
    period_sums[group] = (all_frames.groupby(pd.Grouper(key="time_stamp",
                                                        freq=measure_frequency))
                                    .sum()
                                    .reset_index())


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

# %%
# We're going to drop all of the frames with poor classifier performance
# this is an easy to think about way to do it.
for g in groups:
        for frame in config["frames"]["low_f1"]:
            if frame in period_sums[g].columns:
                period_sums[g] = period_sums[g].drop(
                    frame, axis="columns")

# %% [markdown]
# ok now that the data is loaded we can start looking at the properties of our
# time series and think about how to model it as described above.
# %%

group = "public"
frame = "Policy Prescription and Evaluation"
plot_df = period_sums[group]

# We're going to put the time series and
fig, ax = plt.subplot_mosaic([["A", "A"],
                              ["B", "B"],
                              ["C", "C"],
                              ["D", "D"]], figsize=(7, 10))
# time series plot
ax["A"].plot(plot_df["time_stamp"], plot_df[frame])
ax["A"].set_xlabel("Tweet Time")
ax["A"].set_ylabel("Number of Tweets")
ax["A"].set_title(f"{group} - {frame}")

# Discrete distribution so we can get unbinned frequencies
if normalize:
    ax["B"].hist(plot_df[frame], bins=50)
    ax["B"].set_xlabel("Fraction Cueing Frame in period")
else:
    values, counts = np.unique(plot_df[frame], return_counts=True)
    ax["B"].bar(values, counts, width=1, linewidth=0)
    ax["B"].set_xlabel("Tweets Cueing Frame in period")
ax["B"].set_ylabel("Count")

# autocorrelation and partial autocorrelation
frame_acf = stattools.acf(plot_df[frame])
frame_pacf = stattools.pacf(plot_df[frame])

if normalize:
    viz_lags = 30
else:
    viz_lags = 50

ax["C"].axhline(0, c="grey")
plot_acf(plot_df[frame], ax["C"], lags=viz_lags, title="")
ax["C"].set_xlabel("Lag")
ax["C"].set_ylabel("Autocorrelation")
ax["C"].set_ylim((-0.5, 1.1))

ax["D"].axhline(0, c="grey")
plot_pacf(plot_df[frame], ax["D"], lags=viz_lags, title="")
ax["D"].set_ylim((-0.5, 1.1))
ax["D"].set_xlabel("Lag")
ax["D"].set_ylabel("Partial Autocorrelation")

plt.tight_layout()
plt.savefig("../plots/time_series_analysis/time_series_diagnostic.png")
plt.show()
# %% [markdown]
#
# For the ones I've looked looked at above, It looks to me like lognormal and
# exponential are both reasonable fits. At least in some cases. Maybe
# exponential may fail to explain the tail but I'm not sure how often that is
# true. I think maybe we should think of these as exponential?
#
# It seems pretty clear to me that we definitely have a sort of periodicity of
# ~24 Hrs (3 lags). Otherwise it looks like we have about a 4 lag significant
# autoregressive effect based on mostly zeros following that on the PACF plot.
# With no negative effect or sudden dropoff on the ACF plot I don't think we
# have integration or moving-average effects.
#
# Therefore, I think we should use a simply AR model to model our timeseries.
# Then we can look at the residuals and hopefully we will be ready to do
# granger causality! I'm not sure if the AR model alone can handle the
# periodicity but I guess we'll find out.
#
# %%
if normalize:
    seasonal = False
    lags = 6
    period = 0
else:
    seasonal = True
    lags = 4
    period = 3 
res = ar_model.AutoReg(plot_df[frame], lags=lags).fit()
print(f"AR, No seasonality. AIC: {res.aic}, BIC: {res.bic}")
res2 = ar_model.AutoReg(plot_df[frame], lags=lags, seasonal=seasonal, period=period).fit()
print(f"AR Seasonality. AIC: {res2.aic}, BIC: {res2.bic}")
res_arima = ARIMA(plot_df[frame], order=(lags, 0, 1),
                  seasonal_order=(0, 0, 0, period)).fit()
print(f"SARIMA I(1). AIC: {res_arima.aic}, BIC: {res_arima.bic}")
res_arima = ARIMA(plot_df[frame], order=(lags, 0, 2),
                  seasonal_order=(0, 0, 0, period)).fit()
print(f"SARIMA I(2). AIC: {res_arima.aic}, BIC: {res_arima.bic}")

# %% [markdown]
#
# Ok so it looks like my attempt to specify seasonality was a little bit better
# than when I didn't do that. Let's look at the distribution of residuals to see
# if it looks likr granger causality is going to work. ARIMA was somehow worse
# than the simple AR model
#
# %%
order = ar_model.ar_select_order(
    plot_df[frame], maxlag=21, seasonal=True, period=3)
print(order.ar_lags)
res = ar_model.AutoReg(
    plot_df[frame], lags=order.ar_lags, seasonal=seasonal, period=period).fit()
print(f"AR(6) AIC: {res.aic}, BIC: {res.bic}")

res_ari = ARIMA(plot_df[frame], order=(6, 0, 1)).fit()
print(f"ARI(6, 1) AIC: {res_ari.aic}, BIC: {res_ari.bic}")

# %%
fig, ax = plt.subplots(nrows=3, figsize=(6, 5))

# in sample prediction plot
n_steps = 100
ar_preds = res.predict(0, len(plot_df[frame][:n_steps]))
ari_preds = res_ari.predict(0, len(plot_df[frame][:n_steps]))
ax[0].plot(plot_df[frame][:n_steps], label="data")
ax[0].plot(ar_preds, ls="--", label="AR(6) Model")
ax[0].plot(ari_preds, ls=":", label="ARI(6, 1) Model")

ax[0].legend()
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Number of Tweets")

ax[1].hist(res.resid, bins=100)
ax[1].set_xlabel("AR(6) Residual")
ax[1].set_ylabel("Count")

ax[2].hist(res_ari.resid, bins=100)
ax[2].set_xlabel("ARI(6, 1) Residual")
ax[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(f"../plots/time_series_analysis/model_fit_{group}_{frame}.png")
plt.savefig(f"../plots/time_series_analysis/model_fit_{group}_{frame}.pdf")
plt.show()

# %% [markdown]
#
# Ok so when I let the code select the maximum lag for the AR model we get 21
# steps which is really alot of parameters. I think we should go with the more
# parsimonious max lag of 4 based on the PACF plot.
#
# The next step then is to fit this model for each of our frames from each of
# our groups and build a new dictionary of the residuals for granger causality.
#

# %%
residuals = {k: {} for k in groups}
all_periods = pd.DataFrame(period_sums["congress"]["time_stamp"])

for group in groups:
    for frame_type in frame_types:
        for frame in config["frames"][frame_type]:

            if frame in config["frames"]["low_f1"]:
                continue

            time_series=period_sums[group][["time_stamp", frame]]

            # we want to make sure that all of the times are actually aligned
            time_series_fixed=pd.merge(all_periods, time_series,
                                       on = "time_stamp",
                                       how = "left").fillna(0)

            ar_fit=ar_model.AutoReg(time_series[frame],
                                    lags=4,
                                    seasonal=True,
                                    period=3).fit()
            
            residuals[group][frame] = ar_fit.resid

# %% [markdown]
#
# Ok so we're finally ready to do this thing. We have a little helper function
# that handles the parsing of the granger causality output and handling
#
# %%
from typing import Tuple
def run_granger_causality(pair: Tuple[str, str],
                          data: dict,
                          frame: str) -> dict:
    
    # we're going to return a dict we can use as a dataframe row
    result = {}
    result["source"] = pair[1]
    result["target"] = pair[0]
    result["frame"] = frame
    
    # pull out a dictionary for the data we are looking at
    target_data = data[pair[0]][frame]
    source_data = data[pair[1]][frame]

    gcdf = pd.DataFrame({"source": source_data, "target": target_data}).fillna(0)
    

    if len(gcdf.columns[gcdf.nunique() <= 1]) > 0:
        return {}

    gc_res = grangercausalitytests(gcdf, maxlag=1)

    test_name = "liklihood-ratio"
    p_value = gc_res[1][0]["lrtest"][1]
    test_statistic = gc_res[1][0]["lrtest"][0]
    

    result["test_name"] = test_name
    result["p_value"] = p_value
    result["test_statistic"] = test_statistic

    return result

# ok we're going to use the above function to get granger causality scores
group_pairs = permutations(groups, 2)
df_rows = []

for pair in group_pairs:
    for frame_type in frame_types:
        for frame in config["frames"][frame_type]:
            if frame in config["frames"]["low_f1"]:
                continue
            else:
                df_rows.append(run_granger_causality(pair, 
                                                     residuals, 
                                                     frame))
        
gcdf = pd.DataFrame(df_rows).dropna()
if normalize:
    norm_label = "_normalized"
else:
    norm_label = ""

gcdf.to_csv(f"../data/time_series_output/sample_granger_causality{norm_label}.tsv",
            sep="\t",
            index=False)
print(gcdf)
# %%
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

gcdf_corrected = bonferroni_holm(gcdf, 0.05)
signif = gcdf_corrected[gcdf_corrected["null_rejected"] == True]
signif.to_csv(f"../data/time_series_output/significant_complete_granger{norm_label}.tsv", sep="\t")
# %% [markdown]
# ## What if we dont use the same model across time series
#
# Based on a comment from ceren I want to see how many lags we should consider
# in our AR models for the univariate time series. I will first plot a
# distribution of optimal lags for each of the frame use time series than i
# will rerun the granger causality analysis using each optimal model if we see
# a broad distribution

# %%
optimal_orders = []
best_bics = []
for group in groups:
    for frame_type in frame_types:
        for frame in config["frames"][frame_type]:

            if frame in config["frames"]["low_f1"]:
                continue
    
            
            time_series = period_sums[group][["time_stamp", frame]]
            if np.sum(time_series[frame]) > 0:
                order = ar_model.ar_select_order(
                    time_series[frame], maxlag=21, seasonal=True, period=3)
                
                best_bic = 0
                for lags in order.bic.keys():
                    if order.bic[lags] < best_bic:
                        best_bic = order.bic[lags]

                best_bics.append(best_bic)

                if order.ar_lags:
                    optimal_orders.append(len(order.ar_lags))
                else:
                    optimal_orders.append(0)
                

value, count = np.unique(optimal_orders, return_counts=True)

fig, ax = plt.subplots()
ax.bar(value, count)
ax.set_xlabel("Optimal AR Lags")
ax.set_ylabel("# of single frame time series")

plt.savefig("../plots/time_series_analysis/optimal_ar_lags.png")
plt.savefig("../plots/time_series_analysis/optimal_ar_lags.pdf")
plt.show()
# %%
residuals = {k: {} for k in groups}
all_periods = pd.DataFrame(period_sums["congress"]["time_stamp"])

for group in groups:
    for frame_type in frame_types:
        for frame in config["frames"][frame_type]:

            if frame in config["frames"]["low_f1"]:
                continue

            time_series=period_sums[group][["time_stamp", frame]]

            # we want to make sure that all of the times are actually aligned
            time_series_fixed=pd.merge(all_periods, time_series,
                                       on = "time_stamp",
                                       how = "left").fillna(0)
            
            order = ar_model.ar_select_order(time_series[frame],
                                             maxlag=21,
                                             seasonal=True,
                                             period=3)

            if order.ar_lags:

                selected_lags = order.ar_lags

                ar_fit=ar_model.AutoReg(time_series[frame],
                                        lags=selected_lags,
                                        seasonal=True,
                                        period=3).fit()

                residuals[group][frame] = ar_fit.resid
            
            else:

                residuals[group][frame] = time_series[frame]

# %%
group_pairs = permutations(groups, 2)
df_rows = []

for pair in group_pairs:
    for frame_type in frame_types:
        for frame in config["frames"][frame_type]:
            if frame in config["frames"]["low_f1"]:
                continue
            else:
                df_rows.append(run_granger_causality(pair, 
                                                     residuals, 
                                                     frame))
        
gcdf = pd.DataFrame(df_rows).dropna()
if normalize:
    norm_label = "_normalized"
else:
    norm_label = ""

gcdf.to_csv(f"../data/time_series_output/optimal_granger_causality{norm_label}.tsv",
            sep="\t",
            index=False)

gcdf_corrected = bonferroni_holm(gcdf, 0.05)
signif = gcdf_corrected
signif.to_csv(f"../data/time_series_output/significant_complete_granger{norm_label}_optimal_ar.tsv", sep="\t")
# %%
signif
# %%
