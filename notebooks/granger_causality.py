
# %%
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from itertools import permutations

with open("../workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# %%
# load data
groups = ["congress", "journalists", "trump", "public"]
frame_types = ["generic", "specific"]  # narrative should be in here too

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
            tweet_df = pd.merge(tweet_info, predictions, how="right", on="id_str")
            frames[group][frame_type] = tweet_df
        else:
            # frame predictions
            predictions = pd.read_csv(f"../data/binary_frames/predicted_frames.tsv",
                                      sep="\t")
            
            tweet_info = pd.read_csv("../data/immigration_tweets/public_sample.tsv",
                                     sep="\t")[["id_str", "time_stamp"]]
            frames["public"][frame_type] = pd.merge(tweet_info, predictions, on="id_str")
            

# %%
weekly = pd.date_range(config["dates"]["start"], config["dates"]["end"],
                       freq="W")
weekly
# %%
# will hold keys for groups
weekly_sums = {k: {} for k in groups}

for group in groups:
    for frame_type in frame_types:
        type_rows = []
        df = frames[group][frame_type]
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True)

        weekly_sums[group][frame_type] = (df.groupby(pd.Grouper(key="time_stamp", 
                                                                freq="W"))
                                            .sum()
                                            .reset_index())

# %%
from typing import Tuple
def run_granger_causality(pair: Tuple[str, str],
                          frame_type: str,
                          frame: str,
                          diff=False) -> dict:
    
    # we're going to return a dict we can use as a dataframe row
    result = {}
    result["source"] = pair[1]
    result["target"] = pair[0]
    result["frame_type"] = frame_type
    result["frame"] = frame
    
    # pull out a dictionary for the data we are looking at
    target_data = weekly_sums[pair[0]][frame_type][["time_stamp", frame]]
    source_data = weekly_sums[pair[1]][frame_type][["time_stamp", frame]]

    gcdf = (pd.merge(target_data, source_data, on="time_stamp", how="outer")
              .fillna(0)
              .drop("time_stamp", axis="columns"))
    
    if diff:
        gcdf = gcdf.diff().fillna(0)

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
                                                     frame_type, 
                                                     frame, 
                                                     diff=False))
        
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
# %%