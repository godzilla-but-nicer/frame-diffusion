# %% [markdown]
# # Logistic Regression of frame influence
#
# This notebook contains most of the code for running the regressions we will
# use to estimate the effects of frame exposure on frame usage. We are
# focusing on the general public in this work since we have the most complete
# data for this group.
#
# In the first cell we are loading a bunch of libraries and data. I will note
# in comments what these things are used for where appropriate.
# %%
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm  # does the regression
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datetime import datetime as dt
from datetime import timedelta

import frame_stats as fs
import frame_stats.time_series as ts
import frame_stats.causal_inferrence as ci  # has the functions for setting up regression
import pickle

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
filtered_tweets = fs.filter_users_by_activity(f, 1)
print("tweets loaded")

# FOR TESTING
filtered_tweets = filtered_tweets.sample(frac=1)

# list of all frame names
all_frame_list = config["frames"]["generic"] + config["frames"]["specific"] + config["frames"]["narrative"]

# adjacency list for neighbor lookup
print("loading adjacency list")
with open(paths["mentions"]["adjacency_list"], "r") as fout:
    mention_neighbors = json.loads(fout.read())
print("adjacency list loaded")

print("loading user time series hash")
with open(paths["user_time_series"], "rb") as utspkl:
    user_time_series = pickle.load(utspkl)
print("loaded user time series hash")

# load the features dataset
print("loading features")
features = pd.read_csv(paths["regression"]["features"], sep="\t", dtype={"id_str": str})
features = features.drop_duplicates()
print("features loaded")
# %% [markdown]
# Ok now we need to built the treatment pairs, Basically for each tweet we look
# for frames in the previous day. This is the sort of self exposure treatment.
# This is what we're calling `all_frame_pairs` a tweet and the frames from the day before
# %%
try:
    with open(paths["regression"]["self_influence_pairs"], "rb") as fout:
        all_frame_pairs = pickle.load(fout)
except:
    all_frame_pairs = []
    print("gathering self-influence pairs \n\n")
    for user in tqdm(filtered_tweets["screen_name"].unique()):
        try:
            user_pairs = ci.construct_tweet_self_influence_pairs(user,
                                                                 filtered_tweets,
                                                                 "1D",
                                                                 config)
        except:
            continue

        if user_pairs:
            all_frame_pairs.extend(user_pairs)

    if len(all_frame_pairs) > 0:
        with open(paths["regression"]["self_influence_pairs"], "wb") as fout:
            pickle.dump(all_frame_pairs, fout)
# %%
print("building self-influence dfs \n\n")
self_regression_dfs = {}
for frame in tqdm(all_frame_list):

    rows = []

    for pair in all_frame_pairs:

        # manually add zeros for dates with no frame info
        if len(pair["t"]) == 0:
            exposure = 0
        else:
            exposure = pair["t"][frame].values[0]
        
        cue = pair["t+1"][frame]
        id = pair["t+1"]["id_str"]

        row = {}
        row["cue"] = cue
        row["id_str"] = str(id)
        row["self_exposure"] = exposure

        rows.append(row)

    self_pairs_df = pd.DataFrame(rows)

    self_regression_dfs[frame] = pd.merge(self_pairs_df, features, on="id_str", how="left")


# %%
endog_col = "cue"
exog_cols = ["self_exposure", 
              "is_quote_status", "is_reply", 
              "log_chars", "log_favorites", "log_retweets",
              "is_verified", "log_followers", "log_following", "log_statuses", "ideology", "log_unique_mentions"]

print("running self-influence regressions")
result_dict = {}
for frame in []:#tqdm(all_frame_list):

    clean_frame_df = self_regression_dfs[frame].dropna(subset=[endog_col] + exog_cols)

    response = clean_frame_df[endog_col]
    predictors = sm.add_constant(clean_frame_df[exog_cols])

    model = sm.Logit(endog=response, exog=predictors)

    try:
        result = model.fit(maxiter=100, method="bfgs")
        result_dict[frame] = result
    except:
        print(f"no results for frame: {frame}")
        result = None
        continue

    header_suffix = f"{frame.lower().replace(' ', '_')}_header.csv"
    with open(paths["regression"]["self_output"] + header_suffix, "w") as f_head:
        f_head.write(result.summary().tables[0].as_csv())

    params_suffix = f"{frame.lower().replace(' ', '_')}_table.csv"
    with open(paths["regression"]["self_output"] + params_suffix, "w") as f_body:
        f_body.write(result.summary().tables[1].as_csv())

#with open(paths["regression"]["result_pickles"] + "self_influence.pkl", "wb") as fout:
#    pickle.dump(result_dict, fout)

# %% [markdown]
#
# Now we have to do the alter influence regressions. The functions should be
# partially done and we just have to put things into shape
#
# %%
try:
    with open(paths["regression"]["alter_influence_pairs"], "rb") as fout:
        all_frame_pairs = pickle.load(fout)
except:
    all_frame_pairs = []
    # ok so this is going to be slow as hell
    print("Checking tweet pairs for alter influence")
    for user in tqdm(filtered_tweets["screen_name"].unique()):

        # we gathered a list of neighbors in an earlier cell ;)
        # now we just have to collect the time series here to pass into the function
        # kind of ugly honestly
        alter_ts_list = []
        if user in mention_neighbors:
            for alter in mention_neighbors[user]:
                if alter in user_time_series:
                    alter_ts_list.append(user_time_series[alter])

            frame_pairs = ci.construct_tweet_alter_influence_pairs(user,
                                                                   alter_ts_list,
                                                                   filtered_tweets,
                                                                   "1D",
                                                                   config)
            if frame_pairs:
                all_frame_pairs.extend(frame_pairs)

    if len(all_frame_pairs) > 0:
        with open(paths["regression"]["alter_influence_pairs"], "wb") as fout:
            pickle.dump(all_frame_pairs, fout)

# %%
print("building alter-influence dfs")
alter_regression_dfs = {}
for frame in tqdm(all_frame_list):

    rows = []

    for pair in all_frame_pairs:
        
        # manually add zeros for dates with no frame info
        if len(pair["t"]) == 0:
            exposure = 0
        else:
            exposure = pair["t"][frame].values[0]
        
        cue = pair["t+1"][frame]
        id = pair["t+1"]["id_str"]

        row = {}
        row["cue"] = cue
        row["id_str"] = str(id)
        row["alter_exposure"] = exposure

        rows.append(row)

    alter_pairs_df = pd.DataFrame(rows)

    alter_regression_dfs[frame] = pd.merge(alter_pairs_df, features, on="id_str", how="left").dropna()
# %%

print("running alter-influence regressions")
endog_col = "cue"
exog_cols = ["alter_exposure", 
              "is_quote_status", "is_reply", 
              "log_chars", "log_favorites", "log_retweets",
              "is_verified", "log_followers", "log_following", "log_statuses", "ideology", "log_unique_mentions"]

result_dict = {}
for frame in []: #tqdm(all_frame_list):
    response = alter_regression_dfs[frame][endog_col]
    predictors = sm.add_constant(alter_regression_dfs[frame][exog_cols])

    model = sm.Logit(endog=response, exog=predictors)

    try:
        result = model.fit(maxiter=100, method="bfgs")
        result_dict[frame] = result
    except Exception as e:
        print(f"no results for frame: {frame}")
        print(f"Exception: {e}")
        continue


    header_suffix = f"{frame.lower().replace(' ', '_')}_header.csv"
    with open(paths["regression"]["alter_output"] + header_suffix, "w") as f_head:
        f_head.write(result.summary().tables[0].as_csv())

    params_suffix = f"{frame.lower().replace(' ', '_')}_table.csv"
    with open(paths["regression"]["alter_output"] + params_suffix, "w") as f_body:
        f_body.write(result.summary().tables[1].as_csv())

#with open(paths["regression"]["result_pickles"] + "alter_influence.pkl", "wb") as fout:
#    pickle.dump(result_dict, fout)
# %%
print("running combined alter and self influence regression")

# we have the exposure variables for self and alter we just have to combine
# them with the features to get the combined regressions
endog_col = "cue"
exog_cols = ["alter_exposure", "self_exposure",
              "is_quote_status", "is_reply", 
              "log_chars", "log_favorites", "log_retweets",
              "is_verified", "log_followers", "log_following", "log_statuses", "ideology", "log_unique_mentions"]

result_dict = {}
combined_frame_dfs = {}
for frame in []: #tqdm(all_frame_list):

    combined_frame_dfs[frame] = pd.merge(alter_regression_dfs[frame][["id_str", "alter_exposure"]],
                                         self_regression_dfs[frame], on="id_str", how="right")
    combined_frame_dfs[frame]["alter_exposure"] = combined_frame_dfs[frame]["alter_exposure"].fillna(0)

    clean_regression_df = combined_frame_dfs[frame].dropna(subset = [endog_col] + exog_cols)

    response = clean_regression_df[endog_col]
    predictors = sm.add_constant(clean_regression_df[exog_cols])

    model = sm.Logit(endog=response, exog=predictors)
    
    try:
        result = model.fit(maxiter=100, method="bfgs")
        result_dict[frame] = result
    except Exception as e:
        print(f"no results for frame: {frame}")
        print(f"Exception: {e}")
        continue

    header_suffix = f"{frame.lower().replace(' ', '_')}_header.csv"
    with open(paths["regression"]["combined_output"] + header_suffix, "w") as f_head:
        f_head.write(result.summary().tables[0].as_csv())

    params_suffix = f"{frame.lower().replace(' ', '_')}_table.csv"
    with open(paths["regression"]["combined_output"] + params_suffix, "w") as f_body:
        f_body.write(result.summary().tables[1].as_csv())

#with open(paths["regression"]["result_pickles"] + "combined_influence.pkl", "wb") as fout:
#    pickle.dump(result_dict, fout)



# %%
# we have the combined frame df data from before but we need to incorporate events
# I think I might need to go back to the beginning to do that

print("running combined regression with events")
events = pd.read_csv(paths["events_data"])

endog_col = "cue"
exog_cols = ["alter_exposure", "self_exposure",
              "is_quote_status", "is_reply", 
              "log_chars", "log_favorites", "log_retweets",
              "is_verified", "log_followers", "log_following", "log_statuses", "ideology", "log_unique_mentions"]

events = events[["date"]]
events["complete_date"] = events["date"]
events = events.drop("date", axis="columns")
events["event"] = 1

result_dict = {}

for frame in all_frame_list:
    rdf = combined_frame_dfs[frame].dropna(subset=["year", "month", "date"]).copy()
    rdf["complete_date"] = rdf.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}-{int(x['date'])}", axis=1)

    clean_regression_df = pd.merge(rdf, events, on="complete_date", how="left").dropna(subset=[endog_col] + exog_cols)
    clean_regression_df = clean_regression_df.fillna(0)

    exog_cols.append("event")
    response = clean_regression_df[endog_col]
    predictors = sm.add_constant(clean_regression_df[exog_cols])
    
    save_cols = [endog_col] + exog_cols


    clean_regression_df.to_csv(f"data/regression/full_{frame}.csv", index=False)

    model = sm.Logit(endog=response, exog=predictors)

    try:
        result = model.fit(maxiter=100, method="bfgs")
        result_dict[frame] = result
    except Exception as e:
        print(f"no results for frame: {frame}")
        print(f"Exception: {e}")
        continue

    header_suffix = f"{frame.lower().replace(' ', '_')}_header.csv"
    with open(paths["regression"]["events_output"] + header_suffix, "w") as f_head:
        f_head.write(result.summary().tables[0].as_csv())

    params_suffix = f"{frame.lower().replace(' ', '_')}_table.csv"
    with open(paths["regression"]["events_output"] + params_suffix, "w") as f_body:
        f_body.write(result.summary().tables[1].as_csv())

with open(paths["regression"]["result_pickles"] + "event_influence.pkl", "wb") as fout:
    pickle.dump(result_dict, fout)

combined_frame_dfs["Economic"]

# %%

# %%
