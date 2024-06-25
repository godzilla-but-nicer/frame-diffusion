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
    frame_types = ["generic", "specific", "narrative"]

    # get the user information wrt the tweets
    congress_info = pd.read_csv(paths["congress"]["metadata"], sep="\t")
    all_congress_tweets = pd.read_csv(paths["congress"]["full_tweets"], sep="\t", dtype={"id_str": str})[["id_str", "screen_name"]]
    labelled_congress_tweets = pd.merge(
        all_congress_tweets, congress_info, on="screen_name")


    congress_frame_dfs = []
    for frame_type in frame_types:
        congress_frame_dfs.append(pd.read_csv(
            paths["congress"]["frames"][frame_type], sep="\t", dtype={"id_str": str}))

    # fancy triple merge "one-liner"
    congress_preds = (reduce(lambda l, r: pd.merge(l, r, on="id_str"),
                             congress_frame_dfs).fillna(0))
    congress_preds["id_str"] = congress_preds["id_str"].astype(str)

    times = pd.read_csv(paths["all_frames"], sep="\t", dtype={"id_str": str})[["id_str", "time_stamp"]]
    congress_preds = pd.merge(congress_preds, times, on="id_str")

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


def load_journalist_frames():

    frame_types = ["generic", "specific", "narrative"]
    
    # load data
    # metadata for our journalists from whatever network they came from
    journalist_position = pd.read_csv(paths["journalists"]["metadata"], sep="\t")
    journalist_position["screen_name"] = journalist_position["username"]
    
    # connects media bias fact check source ratings to screen names in our dataset
    journalist_bias_mapper = pd.read_csv(paths["journalists"]["mbfc_mapper"], sep="\t")
    journalist_bias_mapper["site"] = journalist_bias_mapper["top_journo_site"]
    
    journalist_position_with_map = pd.merge(journalist_bias_mapper, journalist_position, on="site")
    

    # contains the media bias fact check bias data
    journalist_bias = pd.read_csv(paths["journalists"]["mbfc"], sep="\t")
    journalist_bias["mbfc_site"] = journalist_bias["site_name"]
    
    journalist_position_and_bias = pd.merge(journalist_bias, journalist_position_with_map, on="mbfc_site")
    
    keep_cols = ["name", "screen_name", "site_name", "bias_rating", "factual_reporting_rating"]
    journalist_info = journalist_position_and_bias[keep_cols]
    
    journalist_info
    
    all_journalist_tweets = pd.read_csv(paths["journalists"]["full_tweets"], sep="\t", dtype={"id_str": str})[["id_str", "screen_name"]]
    labelled_journalist_tweets = pd.merge(
        all_journalist_tweets, journalist_info, on="screen_name", how="left")
    
    journalist_frame_dfs = []
    for frame_type in frame_types:
        journalist_frame_dfs.append(pd.read_csv(
            paths["journalists"]["frames"][frame_type], sep="\t"))
    
    # fancy triple merge "one-liner"
    journalist_preds = (reduce(lambda l, r: pd.merge(l, r, on="id_str"),
                                journalist_frame_dfs).fillna(0))
    journalist_preds["id_str"] = journalist_preds["id_str"].astype(str)

    times = pd.read_csv(paths["all_frames"], sep="\t", dtype={"id_str": str})[["id_str", "time_stamp"]]
    labelled_journalist_tweets = pd.merge(labelled_journalist_tweets, times, on="id_str")

    return pd.merge(labelled_journalist_tweets, journalist_preds, on="id_str").drop(["text_x", "text_y"], axis="columns")


def load_public_frames():

    full_datasheet = pd.read_csv(paths["public"]["metadata"], sep="\t")
    us_datasheet = full_datasheet[full_datasheet["country"] == "US"]
    del full_datasheet

    all_frames = pd.read_csv(paths["public"]["frames"]["all"], sep="\t")[["id_str", "time_stamp"]]
    us_datasheet = pd.merge(us_datasheet, all_frames, on="id_str")
    
    drop_cols = ["Unnamed: 0", "country", "has_hashtag", "has_mention", "has_url",
                "is_quote_status", "is_reply", "is_verified", "log_chars",
                "log_favorites", "log_followers", "log_following",
                "log_retweets", "log_statuses", "month", "year"]
    return us_datasheet.drop(drop_cols, axis="columns")

def load_trump_frames():
    
    trump_frame_dfs = []
    for ft in ["generic", "specific", "narrative"]:
        trump_frame_dfs.append(pd.read_csv(paths["trump"]["frames"][ft], sep="\t"))
    
    trump_preds = (reduce(lambda l, r: pd.merge(l, r, on="id_str"), trump_frame_dfs))
    trump_preds = trump_preds.drop(["text_x", "text_y"], axis="columns")
    
    trump_metadata = pd.read_csv(paths["trump"]["full_tweets"], sep="\t", engine="python")
    trump_metadata = trump_metadata.drop(["text", "year"], axis="columns")
    
    trump = pd.merge(trump_preds, trump_metadata, on="id_str")

    return trump