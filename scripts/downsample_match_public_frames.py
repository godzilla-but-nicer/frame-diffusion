import gzip
import json
import numpy as np
import pandas as pd

public_frames = pd.read_csv("data/binary_frames/predicted_frames.tsv", sep="\t")

keep_ids = []
for year in ["2018", "2019"]:
    for tweet_str in gzip.open(f"data/down_sample/immigration_tweets/US_{year}.gz"):
        tweet = json.loads(tweet_str)
        keep_ids.append(str(tweet["id"]))

id_df = pd.DataFrame({"id_str": keep_ids})

kept_frames = pd.merge(public_frames, id_df, on="id_str")

kept_frames.to_csv(f"data/down_sample/binary_frames/predicted_frames.tsv", sep="\t", index=False)
