import datetime
import json
import requests
import re
import tweet_handler as th
from tqdm import tqdm

# get all of the urls we need
base_url = "https://raw.githubusercontent.com/alexlitel/congresstweets/master/data/"
start_date = datetime.date(2018, 1, 1)
num_days = 365
date_range = [start_date + datetime.timedelta(days=x) for x in range(num_days)]

# make a list of tweets we want to keep based on a query
query = "immigration|immigrants?|illegals|undocumented|illegal aliens?|migrants?|migration"
filtered_tweets = []

# we're going to look day by day to grab all these tweets
for day in tqdm(date_range):
    response = requests.get(base_url + str(day) + ".json")
    tweets = json.loads(response.text)
    filtered_tweets.extend(th.filter_tweet_list(tweets, query))


with open("data/immigration_tweets/congress_2018.json", "w") as fout:
    json.dump(filtered_tweets, fout)
