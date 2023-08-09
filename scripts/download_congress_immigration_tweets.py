import datetime
import json
import requests
import re
import tweet_handler as th
import pandas as pd
from tqdm import tqdm

# get all of the urls we need
base_url = "https://raw.githubusercontent.com/alexlitel/congresstweets/master/data/"
start_date = th.start_date
end_date = th.end_date

# make a list of tweets we want to keep based on a query
query = th.immigration_keywords
filtered_tweets = []
filter_rejects = []

# we're going to look day by day to grab all these tweets
day_range = pd.date_range(start_date, end_date, freq="1D")
for day in tqdm(day_range):
    response = requests.get(base_url + str(day.date()) + ".json")
    tweets = json.loads(response.text)
    imm_tweets, gen_tweets = th.filter_tweet_list(tweets, query, return_rejects=True)
    filtered_tweets.extend(imm_tweets)
    filter_rejects.extend(gen_tweets)

with open("data/immigration_tweets/congress.json", "w") as fout:
    json.dump(filtered_tweets, fout)

with open("data/non_immigration_tweets/congress.json", "w") as fout:
    json.dump(filter_rejects, fout)