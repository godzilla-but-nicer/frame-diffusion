import json
import pandas as pd

from functools import reduce

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())


groups = ["congress", "journalists", "trump", "public"]
frame_types = ["generic", "specific", "narrative"]
all_frame_dfs = []

frames = {k: {} for k in groups}
for group in groups:
    type_dfs = []
    if group != "public":
        for frame_type in frame_types:

            # all frames are in one file for the public
            # we have to merge a couple of different files in either case
            if group != "public":
                # predicted frames
                predictions = pd.read_csv(paths[group]["frames"][frame_type],
                                            sep="\t", dtype={"id_str": str}).drop("text", axis="columns")

                # time stamps for granger causality
                tweet_info = pd.read_csv(paths[group]["full_tweets"],
                                            sep="\t", dtype={"id_str": str})[["id_str", "time_stamp", "screen_name"]]

                # merge and add to list of dataframes to combine
                tweet_df = pd.merge(tweet_info, predictions,
                                    how="right", on="id_str")
                tweet_df["group"] = group
                type_dfs.append(tweet_df)

        all_frames_for_group = reduce(lambda l, r: pd.merge(l, r,
                                                            on=["id_str",
                                                                "time_stamp",
                                                                "screen_name",
                                                                "group"]),
                                        type_dfs)

    else:
        # frame predictions
        predictions = pd.read_csv(paths["public"]["frames"]["all"],
                                    sep="\t", dtype={"id_str": str})
        tweet_info = pd.read_csv("data/us_public_ids.tsv", sep="\t", dtype={"id_str": str})
        tweet_info["time_stamp"] = pd.to_datetime(tweet_info["time_stamp"])

        # tweet_info["time_stamp"] = datetime.strptime(tweet_info["time_stamp"],
        #                                              "%a %b %d %H:%M:%S +0000 %Y")

        all_frames_for_group = pd.merge(tweet_info, predictions, on="id_str")
        all_frames_for_group["group"] = "public"

    all_frame_dfs.append(all_frames_for_group)

    all_frames = pd.concat(all_frame_dfs)

# get all of the valid frame labels
all_frame_labels = []
for frame_type in ["generic", "specific", "narrative"]:
    all_frame_labels.extend(config["frames"][frame_type])

keep_cols = ["id_str", "time_stamp", "screen_name", "group"] + all_frame_labels

all_frames = all_frames[keep_cols]

all_frames.to_csv(paths["all_frames"], sep="\t", index=False)