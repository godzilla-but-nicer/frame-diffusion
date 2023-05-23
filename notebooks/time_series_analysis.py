# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa import ar_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import stattools
from scipy.stats import lognorm, poisson, expon
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import permutations

with open("../workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# %%
# load data
groups = ["congress", "journalists", "trump", "public"]
# narrative should be in here too
frame_types = ["generic", "specific", "narrative"]

frames = {k: {} for k in groups}
for group in groups:
    for frame_type in frame_types:

        # all frames are in one file for the public
        # we have to merge a couple of different files in either case
        if group != "public":
            # predicted frames
            predictions = pd.read_csv(f"../data/binary_frames/{group}/{group}_{frame_type}.tsv",
                                      sep="\t").drop("text", axis="columns")

            # time stamps for granger causality
            tweet_info = pd.read_csv(f"../data/immigration_tweets/{group}.tsv",
                                     sep="\t")[["id_str", "time_stamp"]]

            tweet_info["id_str"] = tweet_info["id_str"].astype(str)
            predictions["id_str"] = predictions["id_str"].astype(str)

            # merge and add to list of dataframes to combine
            tweet_df = pd.merge(tweet_info, predictions,
                                how="right", on="id_str")
            frames[group][frame_type] = tweet_df
        else:
            # frame predictions
            predictions = pd.read_csv(f"../data/binary_frames/predicted_frames.tsv",
                                      sep="\t")

            tweet_info = pd.read_csv("../data/us_public_ids.tsv", sep="\t")
            tweet_info["time_stamp"] = pd.to_datetime(tweet_info["time_stamp"])
            
            # tweet_info["time_stamp"] = datetime.strptime(tweet_info["time_stamp"],
            #                                              "%a %b %d %H:%M:%S +0000 %Y")
            
            frames["public"][frame_type] = pd.merge(
                tweet_info, predictions, on="id_str")


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
# eight-hour windows
measure_frequency = "8H"


# will hold keys for groups
period_sums = {k: {} for k in groups}

for group in groups:
    for frame_type in frame_types:
        type_rows = []
        df = frames[group][frame_type]
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True)

        period_sums[group][frame_type] = (df.groupby(pd.Grouper(key="time_stamp",
                                                                freq=measure_frequency))
                                            .sum()
                                            .reset_index())

# %%
# We're going to drop all of the frames with poor classifier performance
# this is an easy to think about way to do it.
for g in groups:
    for ft in frame_types:
        for frame in config["frames"]["low_f1"]:
            if frame in period_sums[g][ft].columns:
                period_sums[g][ft] = period_sums[g][ft].drop(
                    frame, axis="columns")

# %% [markdown]
# ok now that the data is loaded we can start looking at the properties of our
# time series and think about how to model it as described above.
# %%

group = "public"
frame_type = "generic"
frame = "Security and Defense"
plot_df = period_sums[group][frame_type]

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
value, count = np.unique(plot_df[frame], return_counts=True)

# let's also fit a couple of distributions
lnshape, lnloc, lnscale = lognorm.fit(plot_df[frame])
eloc, escale = expon.fit(plot_df[frame])

ax["B"].bar(value, count / np.sum(count),
            linewidth=0,
            width=1,
            align="center",
            label="Data")
ax["B"].plot(value[1:], lognorm.pdf(value[1:], lnshape, lnloc, lnscale),
             c="C1",
             ls="--",
             label="Lognormal fit")
ax["B"].plot(value, poisson.pmf(value, np.mean(plot_df[frame])),
             c="C2",
             ls="--",
             label="Poisson fit")
ax["B"].plot(value, expon.pdf(value, eloc, escale),
             c="C3",
             ls="--",
             label="Exponential fit")


ax["B"].legend()
ax["B"].set_xlabel("Number of Tweets")
ax["B"].set_ylabel("Frequency")
ax["B"].set_xlim(-0.5, 53)

# autocorrelation and partial autocorrelation
frame_acf = stattools.acf(plot_df[frame])
frame_pacf = stattools.pacf(plot_df[frame])

ax["C"].axhline(0, c="grey")
plot_acf(plot_df[frame], ax["C"], lags=50, title="")
ax["C"].set_xlabel("Lag")
ax["C"].set_ylabel("Autocorrelation")
ax["C"].set_ylim((-0.5, 1.1))

ax["D"].axhline(0, c="grey")
plot_pacf(plot_df[frame], ax["D"], lags=50, title="")
ax["D"].set_ylim((-0.5, 1.1))
ax["D"].set_xlabel("Lag")
ax["D"].set_ylabel("Partial Autocorrelation")

plt.tight_layout()
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

res = ar_model.AutoReg(plot_df[frame], lags=4).fit()
print(f"AR, No seasonality. AIC: {res.aic}, BIC: {res.bic}")
res2 = ar_model.AutoReg(plot_df[frame], lags=4, seasonal=True, period=3).fit()
print(f"AR Seasonality. AIC: {res2.aic}, BIC: {res2.bic}")
res_arima = ARIMA(plot_df[frame], order=(4, 0, 0),
                  seasonal_order=(0, 0, 0, 3)).fit()
print(f"SARIMA. AIC: {res_arima.aic}, BIC: {res_arima.bic}")

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
    plot_df[frame], lags=order.ar_lags, seasonal=True, period=3).fit()
print(f"AIC: {res.aic}, BIC: {res.bic}")

# %%
fig, ax = plt.subplots(nrows=2, figsize=(6, 4))

# in sample prediction plot
n_steps = 100
preds = res.predict(0, len(plot_df[frame][:n_steps]))
ax[0].plot(plot_df[frame][:n_steps], label="data")
ax[0].plot(preds, label="AR Model")

ax[0].legend()
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Number of Tweets")

ax[1].hist(res.resid, bins=100)

ax[1].set_xlabel("Residual")
ax[1].set_ylabel("Count")
plt.tight_layout()
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
all_periods = pd.DataFrame(period_sums["congress"]["specific"]["time_stamp"])

for group in groups:
    for frame_type in frame_types:
        for frame in config["frames"][frame_type]:

            if frame in config["frames"]["low_f1"]:
                continue

            time_series=period_sums[group][frame_type][["time_stamp", frame]]

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

    max_pvalue = 0
    test_statistic = 0
    test_name = ""
    for test in gc_res[1][0].keys():
        if gc_res[1][0][test][1] > max_pvalue:
            max_pvalue = gc_res[1][0][test][1]
            test_statistic = gc_res[1][0][test][0]
            test_name = test
    

    result["test_name"] = test_name
    result["p_value"] = max_pvalue
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
gcdf.to_csv("../data/time_series_output/sample_granger_causality.tsv",
            sep="\t",
            index=False)
print(gcdf)
# %%
alpha = 0.05

# multiple testing proceedure
def bonferroni_holm(data, alpha):
    sorted = data.sort_values("p_value")
    rejections = np.zeros(sorted.shape[0], dtype=bool)

    for row_i in range(sorted.shape[0]):

        # functional alpha for each iteration
        abh = alpha / (sorted.shape[0] - row_i)

        if sorted.iloc[row_i]["p_value"] < abh:
            rejections[row_i] = True
        else:
            break

    sorted["null_rejected"] = rejections
    return sorted

gcdf_corrected = bonferroni_holm(gcdf, 0.05)
signif = gcdf_corrected[gcdf_corrected["null_rejected"] == True]
signif.to_csv("../data/time_series_output/significant_complete_granger.tsv", sep="\t")
# %%