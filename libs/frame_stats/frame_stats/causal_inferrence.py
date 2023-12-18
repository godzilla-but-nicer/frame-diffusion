import numpy as np
import pandas as pd
from typing import List, Dict, Optional

import frame_stats.time_series as ts

def construct_frame_pairs(user_time_series: pd.DataFrame,
                          time_delta_ammount: int = 1,
                          time_delta_unit: str = "days") -> List[Dict]:

    day_pairs = []
    delta_t = pd.Timedelta(time_delta_ammount, time_delta_unit)

    for t, day in user_time_series.iterrows():

        # any cued frames in consecutive days
        if t + delta_t in user_time_series.index:
            if (user_time_series.loc[t].sum() > 0 and
                user_time_series.loc[t + delta_t].sum() > 0):
    
                pair = {}
                pair["t"] = user_time_series.loc[t]
                pair["t+1"] = user_time_series.loc[t + delta_t]
                pair["t+1_date"] = t + delta_t
    
                day_pairs.append(pair)
    
    return day_pairs


def construct_tweet_self_influence_pairs(user: str,
                                        tweets: pd.DataFrame,
                                        time_delta: str,
                                        config: Dict) -> Optional[List[Dict]]:
    
    user_time_series_df = ts.construct_frame_time_series(tweets,
                                                         user,
                                                         time_delta,
                                                         config)
    
    user_frame_pairs = []
    delta_t = pd.Timedelta(time_delta)  # for looking backward in time
    for _, tweet in tweets[tweets["screen_name"] == user].iterrows():

        # identify dates and frames for time t
        previous_day = pd.to_datetime(tweet["time_stamp"]).date() - delta_t
        past_frames = user_time_series_df.loc[pd.to_datetime(previous_day, utc=True)]

        # if we have frames in time t, we can build an observation
        if past_frames.sum() > 0:
            tweet_pair = {}
            tweet_pair["t"] = past_frames
            tweet_pair["t+1"] = tweet
            user_frame_pairs.append(tweet_pair)

    
    if len(user_frame_pairs) > 0:
        return user_frame_pairs
    else:
        return None



def construct_influencer_frame_pairs(user_time_series: pd.DataFrame,
                                     influencer_time_series: List[pd.DataFrame],
                                     time_delta_ammount: 1,
                                     time_delta_unit: "days") -> List[Dict]:
    
    day_pairs = []
    delta_t = pd.Timedelta(time_delta_ammount, time_delta_unit)

    for t, day_frames in user_time_series.iterrows():

        # sum influencer cued frames for day t
        influencer_frame_array = np.zeros(day_frames.values.shape)
        for its in influencer_time_series:
            if t in its["time_series"].index:
                influencer_frame_array += its["time_series"].loc[t].values

        # convert to series
        influencer_day_frames = pd.Series(influencer_frame_array, index=user_time_series.columns, dtype=float)
        
        # then its essentially the same proceedure as before
        if t + delta_t in user_time_series.index:
            if (influencer_day_frames.sum() > 0 and
                user_time_series.loc[t + delta_t].sum() > 0):

                
                pair = {}
                pair["t"] = influencer_day_frames
                pair["t+1"] = user_time_series.loc[t + delta_t]

                day_pairs.append(pair)


    return day_pairs

def construct_tweet_alter_influence_pairs(user: str,
                                          alter_time_series: List[pd.DataFrame],
                                          tweets: pd.DataFrame,
                                          time_delta: str,
                                          config: Dict) -> Optional[List[Dict]]:

    if len(alter_time_series) < 1:
        return None

    # we need to aggregate all of the alter time series into one
    ts_index = alter_time_series[0].index
    ts_cols = alter_time_series[0].columns
    ts_empty_array = np.zeros(alter_time_series[0].values.shape)

    for _, alter_ts in enumerate(alter_time_series):
        if alter_ts.values.shape == ts_empty_array.shape:
            ts_empty_array += alter_ts.values
    
    combined_alter_time_series = pd.DataFrame(ts_empty_array,
                                              index=ts_index,
                                              columns=ts_cols)


    user_frame_pairs = []
    delta_t = pd.Timedelta(time_delta)  # for looking backward in time
    for _, tweet in tweets[tweets["screen_name"] == user].iterrows():

        # identify dates and frames for time t
        previous_day = pd.to_datetime(tweet["time_stamp"]).date() - delta_t
        past_frames = combined_alter_time_series.loc[pd.to_datetime(previous_day, utc=True)]

        # if we have frames in time t, we can build an observation
        if past_frames.sum() > 0:
            tweet_pair = {}
            tweet_pair["t"] = past_frames
            tweet_pair["t+1"] = tweet
            user_frame_pairs.append(tweet_pair)

    if len(user_frame_pairs) > 0:
        return user_frame_pairs
    else:
        return None
