import datetime
import json
import requests
import re
import tweet_handler as th
from tqdm import tqdm

# get all of the urls we need
base_url = "https://raw.githubusercontent.com/alexlitel/congresstweets/master/data/"
start_date = th.start_date
end_date = th.end_date

# make a list of tweets we want to keep based on a query
query = th.immigration_keywords
filtered_tweets = []

# we're going to look day by day to grab all these tweets
day = start_date.date
while day <= end_date.date:
    response = requests.get(base_url + str(day) + ".json")
    tweets = json.loads(response.text)
    filtered_tweets.extend(th.filter_tweet_list(tweets, query))
    day += datetime.time(days=1)

with open("data/immigration_tweets/congress.json", "w") as fout:
    json.dump(filtered_tweets, fout)
