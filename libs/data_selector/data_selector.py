import numpy as np
import pandas as pd
import json
from functools import reduce


with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

with open("workflow/paths.json", "r") as pf:
    paths = json.loads(pf.read())

def load_congress_frames() -> pd.DataFrame:

    # load data
    frame_sums = {}  # will hold a bunch of vectors of frame sums by group
    frame_types = ["generic", "specific", "narrative"]

    # get the user information wrt the tweets
    congress_info = pd.read_csv(paths["congress"]["metadata"], sep="\t")
    all_congress_tweets = pd.read_csv(paths["congress"]["full_tweets"], sep="\t")[["id_str", "screen_name"]]
    labelled_congress_tweets = pd.merge(
        all_congress_tweets, congress_info, on="screen_name")


    congress_frame_dfs = []
    for frame_type in frame_types:
        congress_frame_dfs.append(pd.read_csv(
            paths["congress"]["frames"][frame_type], sep="\t"))

    # fancy triple merge "one-liner"
    congress_preds = (reduce(lambda l, r: pd.merge(l, r, on="id_str"),
                             congress_frame_dfs).fillna(0))
    congress_preds["id_str"] = congress_preds["id_str"].astype(str)

    return _congress_affiliations(labelled_congress_tweets, congress_preds)


def _congress_affiliations(labelled_congress_tweets, congress_preds) -> pd.DataFrame:

    # conditionals for sorting
    dem_label = labelled_congress_tweets["party"] == "D"
    dem_in_name = labelled_congress_tweets["screen_name"].str.lower().str.contains("dem")
    gop_label = labelled_congress_tweets["party"] == "R"
    gop_in_name = labelled_congress_tweets["screen_name"].str.lower().str.contains("gop")
    not_explicit = (labelled_congress_tweets["party"].isna())

    # now we want to split the predictions by political affiliation
    dem_info = labelled_congress_tweets[dem_label | (dem_in_name & not_explicit)]
    gop_info = labelled_congress_tweets[gop_label | (gop_in_name & not_explicit)]
    indep_info = labelled_congress_tweets[(labelled_congress_tweets["party"] == "I") |
                                            (labelled_congress_tweets["party"] == "L")]
    bipar_info = labelled_congress_tweets[not_explicit &  ~dem_in_name & ~gop_in_name]

    # combine the info with the predicted frames
    dem_pred = pd.merge(dem_info, congress_preds, on="id_str")
    dem_pred["Affiliation"] = "Democrat"
    gop_pred = pd.merge(gop_info, congress_preds, on="id_str")
    gop_pred["Affiliation"] = "Republican"
    indep_pred = pd.merge(indep_info, congress_preds, on="id_str")
    indep_pred["Affiliation"] = "Independent"
    bipar_pred = pd.merge(bipar_info, congress_preds, on="id_str")
    bipar_pred["Affiliation"] = "Bipartisan"

    aff_df = pd.concat([dem_pred, gop_pred, indep_pred, bipar_pred])

    aff_clean = aff_df.drop(["Unnamed: 0", "text_x", "text_y"], axis="columns")
    return aff_clean

