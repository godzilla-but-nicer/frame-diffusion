import json
import tweepy
from datetime import datetime

start = datetime.fromisoformat("2019-01-01T00:00:00")
end = datetime.fromisoformat("2019-01-01T01:00:00")

query = "ping"

with open("workflow/twitter_api_keys.json", "r") as fin:
    keys = json.loads(fin.read())

client = tweepy.Client(keys["bearer_token"])

print(client.search_all_tweets(query=query,
                               start_time=start,
                               end_time=end))
