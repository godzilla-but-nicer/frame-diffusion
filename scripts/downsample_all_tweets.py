import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import tweet_handler as th

fraction = 0.1

# public
# first sample the json
public_sample = [[], []]
for i, year in enumerate(["2018", "2019"]):
    for tweet_str in tqdm(gzip.open(f"data/immigration_tweets/US_{year}.gz")):
        if np.random.uniform() < fraction:
            tweet = json.loads(tweet_str)
            public_sample[i].append(tweet)

# save it
    with gzip.open(f"data/down_sample/immigration_tweets/US_{year}.gz", "wt") as f:
        f.write(json.dumps(public_sample[i]))

# journalists
# sample json
journalist_sample = []
with open("data/immigration_tweets/journalists.json", "r") as fin:
    j_tweets = json.loads(fin.read())
    for tweet in j_tweets:
        if np.random.uniform() < fraction:
            journalist_sample.append(tweet)

# save as json
with open("data/down_sample/immigration_tweets/journalists.json", "w") as fout:
    json.dump(journalist_sample, fout)

df_rows = []
for tweet in journalist_sample:
    row = th.parse_tweet_json(tweet, "v2")
    df_rows.append(row)

df = pd.DataFrame(df_rows)
with open("data/down_sample/immigration_tweets/journalists.tsv", "w") as tsv_file:
    df.to_csv(tsv_file, sep="\t", index=False)


# congress
congress_sample = []
with open("data/immigration_tweets/congress.json", "r") as fin:
    c_tweets = json.loads(fin.read())
    for tweet in c_tweets:
        if np.random.uniform() < fraction:
            congress_sample.append(tweet)

# save json
with open("data/down_sample/immigration_tweets/congress.json", "w") as fout:
    json.dump(congress_sample, fout)

# save tsv
df_rows = []
for tweet in congress_sample:
    row = th.parse_tweet_json(tweet, "congress")
    df_rows.append(row)

df = pd.DataFrame(df_rows)
with open("data/down_sample/immigration_tweets/congress.tsv", "w") as tsv_file:
    df.to_csv(tsv_file, sep="\t", index=False)

# trump
trump_sample = []
with open("data/immigration_tweets/trump.json", "r") as fin:
    t_tweets = json.loads(fin.read())
    for tweet in t_tweets:
        if np.random.uniform() < fraction:
            trump_sample.append(tweet)

with open("data/down_sample/immigration_tweets/trump.json", "w") as fout:
    json.dump(trump_sample, fout)

# save tsv
df_rows = []
for tweet in trump_sample:
    row = th.parse_tweet_json(tweet, "trump")
    df_rows.append(row)

df = pd.DataFrame(df_rows)
with open("data/down_sample/immigration_tweets/trump.tsv", "w") as tsv_file:
    df.to_csv(tsv_file, sep="\t", index=False)


# retweets
for year in ["2018", "2019"]:
    yearly_retweets = pd.read_csv(f"data/decahose_retweets/{year}_retweets.tsv", sep="\t", nrows=10000)
    sample_retweets = yearly_retweets.sample(frac=fraction)
    sample_retweets.to_csv(f"data/down_sample/decahose_retweets/{year}_retweets.tsv", sep="\t", index=False)
