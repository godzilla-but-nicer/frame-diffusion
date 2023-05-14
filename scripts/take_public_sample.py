import json
import numpy as np
import pandas as pd
import tweet_handler as th

# number of tweets to use in our sample
sample_size = 30000

# all classified tweet ids
public_ids = pd.read_csv("data/us_public_ids.tsv",
                         sep="\t")["id_str"]

# ids of the sampled tweets
sample_ids = np.random.choice(public_ids, size=sample_size, replace=False)
sample = pd.DataFrame({"id_str": sample_ids})


sample_tweets = th.load_tweets_from_gz("data/immigration_tweets/US.gz",
                                       keep_ids=set(sample["id_str"].astype(str)))

df_rows = []
for tweet in sample_tweets:
    df_rows.append(th.parse_tweet_json(tweet, "v1"))

sample_tweet_info = pd.DataFrame(df_rows)
sample_tweet_info.to_csv("data/immigration_tweets/public_sample.tsv",
                         index=False, sep="\t")
