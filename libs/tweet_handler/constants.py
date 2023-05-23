import json
from datetime import datetime

with open("workflow/config.json", "r") as cf:
    config = json.loads(cf.read())

# we have to refer to these a bunch so lets put them in one place
immigration_keywords = config["regex_query"]

# these are the dates of the public tweets data
start_date = datetime.strptime(config["dates"]["start"], "%m-%d-%Y %H:%M:%S")
end_date = datetime.strptime(config["dates"]["end"], "%m-%d-%Y %H:%M:%S")

# all fields we will ever need
api_tweet_fields = ["id", "created_at", "text", "public_metrics", 
                    "in_reply_to_user_id", "entities", "referenced_tweets"]
api_tweet_expansions = ["author_id", "referenced_tweets.id"]
api_user_fields = ["id", "name", "username"]