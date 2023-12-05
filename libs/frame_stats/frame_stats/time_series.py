import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# core function for extracting the frame time series. used in the more complex
# functions below
def construct_frame_time_series(df: pd.DataFrame,
                                user: str,
                                frequency: str,
                                config: Dict) -> pd.DataFrame:

    # pull out the data for just our focal user
    user_df = df[df["screen_name"] == user]

    # add empty rows with time stamps for our earliest and latest time stamps
    earliest_row = pd.DataFrame({"screen_name": [user],
                                 "time_stamp": [config["dates"]["start"]]})
    latest_row = pd.DataFrame({"screen_name": [user],
                               "time_stamp": [config["dates"]["end"]]})
    user_df_modified = pd.concat((user_df, earliest_row, latest_row))
    
    # ensure that the time stamp column is being stored correctly
    user_df_modified["time_stamp"] = pd.to_datetime(user_df_modified["time_stamp"],
                                                    utc=True)


    # ok now we can build the time series
    user_df_frames = user_df_modified.drop(["id_str", "screen_name", "group"],
                                           axis="columns")

    user_ts = (user_df_frames.groupby(pd.Grouper(key="time_stamp",
                                                 freq=frequency))
                      .sum()
                      .reset_index())
    
    return user_ts.set_index("time_stamp")

# this really is not about time series per se but is important to identify
# neighbors in the mention network effectively
def longer_mention_subset(mention_subset: pd.DataFrame) -> pd.DataFrame:
    first_half_cols = ["uid1", "uid2", "1to2freq"]
    first_half = mention_subset[first_half_cols]
    first_half["source"] = first_half["uid1"]
    first_half["target"] = first_half["uid2"]
    first_half["weight"] = first_half["1to2freq"]
    first_half = first_half[["source", "target", "weight"]]

    second_half_cols = ["uid1", "uid2", "2to1freq"]
    second_half = mention_subset[second_half_cols]
    second_half["source"] = second_half["uid2"]
    second_half["target"] = second_half["uid1"]
    second_half["weight"] = second_half["2to1freq"]
    second_half = second_half[["source", "target", "weight"]]

    return pd.concat((first_half, second_half))

# little helpers to convert screen name to user id and vice versa
def get_screen_name(user_id, id_map: pd.DataFrame) -> Optional[str]:

    isolated_user = id_map[id_map["user_id"] == user_id]["screen_name"]
    
    if isolated_user.shape[0] > 0:
        return isolated_user.values[0]
    else:
        return None



def get_user_id(screen_name, id_map: pd.DataFrame) -> str:

    isolated_user = id_map[id_map["screen_name"] == screen_name]["user_id"]

    if isolated_user.shape[0] > 0:
        return isolated_user.values[0]
    else:
        return None

# uses mention network to get potential influencers
def get_mentioned_screen_names(screen_name: str,
                               mentions: pd.DataFrame,
                               id_map: pd.DataFrame) -> dict:
    
    user_id = get_user_id(screen_name, id_map)

    # skip this user if we can't find id
    if not user_id:
        return
    
    user_ego = mentions[(mentions["uid1"] == user_id) | (mentions["uid2"] == user_id)]
    user_ego_longer = longer_mention_subset(user_ego)

    user_is_source = user_ego_longer[user_ego_longer["source"] == user_id]
    
    influencers = []
    for i, alter in user_is_source.iterrows():

        alter_screen_name = get_screen_name(alter["target"], id_map)
        
        # same for the alters
        if not alter_screen_name:
            continue

        alter_weight = alter["weight"]

        influencers.append({"screen_name": alter_screen_name, "weight": alter_weight})

    return influencers

# gets time series for each influencer of a user
def get_influencer_time_series(focal_user: str,
                               tweets: pd.DataFrame,
                               config: Dict,
                               mentions: pd.DataFrame,
                               id_map: pd.DataFrame) -> List:
    influencers = get_mentioned_screen_names(focal_user, mentions, id_map)
    
    influencer_ts_list = []
    if influencers:
        for influencer in influencers:
    
            influencer_ts_info = {}
            influencer_ts_df = construct_frame_time_series(tweets, influencer["screen_name"], "1D", config)
            
            influencer_ts_info["screen_name"] = influencer["screen_name"]
            influencer_ts_info["weight"] = influencer["weight"]
            influencer_ts_info["time_series"] = influencer_ts_df
    
            influencer_ts_list.append(influencer_ts_info)
    
    return influencer_ts_list