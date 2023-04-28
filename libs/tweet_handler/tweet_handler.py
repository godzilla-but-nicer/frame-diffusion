import gzip
import json
import re
import tweepy
import time
from constants import *
from datetime import datetime
from typing import List, Optional


# modified from https://github.com/juliamendelsohn/framing/blob/master/code/parse_tweets.py
def get_tweet_text(obj):
    if 'text' not in obj and 'extended_tweet' not in obj:
        return None
    if 'extended_tweet' in obj:
        tweet_text = obj['extended_tweet']['full_text']
    else:
        tweet_text = obj['text']
    return tweet_text.replace('\t', ' ').replace('\n', ' ')


def filter_by_text(tweet_json: dict, query: str) -> Optional[dict]:

    # first ensure that the tweet has text, i guess some dont??
    text = get_tweet_text(tweet_json)

    # if we find the query return the tweet
    if text is None:
        return None
    elif re.search(query, text):
        return tweet_json
    else:
        return None


def filter_tweet_list(tweet_list: List[dict], text_query: str) -> List[dict]:

    # just applying our the filter by text function to a bunch of tweets
    keep_tweets = []

    for tweet in tweet_list:
        filtered_tweet = filter_by_text(tweet, text_query)

        if filtered_tweet is not None:
            keep_tweets.append(filtered_tweet)

    return keep_tweets


def download_immigration_tweets(screen_name: str,
                                start_date: datetime,
                                end_date: datetime,
                                api_bearer_token: str) -> List[dict]:
    
    # set up the client with the api keys
    client = tweepy.Client(api_bearer_token,
                           wait_on_rate_limit=True)

    api_query = build_user_immigration_query(screen_name)

    user_immigration_tweets = []

    for tweet in tweepy.Paginator(client.search_all_tweets,
                                        query=api_query,
                                        start_time=start_date,
                                        end_time=end_date,
                                        tweet_fields=api_tweet_fields,
                                        user_fields=api_user_fields,
                                        expansions=api_tweet_expansions,
                                        limit=50).flatten():
    
        tweet.data["screen_name"] = screen_name
        user_immigration_tweets.append(tweet.data)
        time.sleep(1)

    return user_immigration_tweets


def build_user_immigration_query(screen_name: str,
                                 regex_keywords: str = immigration_keywords):

    # we need to reformat the keywords for the twitter api
    api_keywords = []
    words = immigration_keywords.split("|")

    for word in words:
        # i could have just done this by hand wow
        # instead im tough and unpacked it in *code*
        m = re.match(r"([a-z ]+)s\?", word)
        if m:
            singular = m.group(1)
            api_keywords.append(singular)
            api_keywords.append(singular + "s")
        else:
            api_keywords.append(word)

    explicit_or_keywords = " OR ".join(api_keywords)

    api_query = "(" + explicit_or_keywords + f") from:{screen_name} lang:en"

    return api_query


def parse_tweet_json(tweet_json: dict, source: str="v2") -> dict:
    # this ugly ass function will take a fancy api dict and turn it
    # into a flat dict with appropriate column names. We need a bunch of
    # versions because not all tweets ceom from our api requests
    
    if source == "trump":
        new_row = parse_tweet_json_trump(tweet_json)

    elif source == "v2":
        new_row = parse_tweet_json_v2(tweet_json)
    
    elif source == "congress":
        new_row = parse_tweet_json_congress(tweet_json)

    else:
        return ValueError(f"No source '{source}' implemented for parse_tweet_json")

    return new_row

def parse_tweet_json_trump(tweet_json: dict) -> dict:
    new_row = {}

    new_row["id_str"] = tweet_json["id"]

    # datetime stuff
    dt = datetime.strptime(tweet_json["date"], "%Y-%m-%d %H:%M:%S")
    new_row["year"] = dt.year
    new_row["time_stamp"] = dt

    new_row["screen_name"] = "realDonaldTrump"
    new_row["text"] = tweet_json["text"]

    new_row["favorite_count"] = tweet_json["favorites"]
    new_row["retweet_count"] = tweet_json["retweets"]

    # not sure exactly how to handle this
    if tweet_json["isRetweet"] == "t":
        new_row["retweet_of_id"] = "unknown"
    
    return new_row


def parse_tweet_json_v2(tweet_json: dict) -> dict:
    # we must build up the rows by unpacking the json dict
    new_row = {}

    # main tweet features. every tweet should have these
    new_row["id_str"] = tweet_json["id"]
    
    time_stamp = datetime.strptime(tweet_json["created_at"], "%Y-%m-%dT%H:%M:%S.000Z")
    new_row["year"] = time_stamp.year
    new_row["time_stamp"] = time_stamp
    new_row["screen_name"] = tweet_json["screen_name"]

    new_row["text"] = tweet_json["text"].replace("\t", " ").replace("\n", " ")

    # metrics. every tweet should also have these
    metrics = tweet_json["public_metrics"]

    new_row["quote_count"] = metrics["quote_count"]
    new_row["reply_count"] = metrics["reply_count"]
    new_row["favorite_count"] = metrics["like_count"]
    new_row["retweet_count"] = metrics["retweet_count"]

    # interactions with other tweets, not all tweets will have this
    if "referenced_tweets" in tweet_json:

        for ref_tw in tweet_json["referenced_tweets"]:
            if ref_tw["type"] == "replied_to":
                new_row["reply_to_id"] = ref_tw["id"]

            if ref_tw["type"] == "retweeted":
                new_row["retweet_of_id"] = ref_tw["id"]
        
            if ref_tw["type"] == "quoted":
                new_row["quote_of_id"] = ref_tw["id"]
    
    return new_row


def parse_tweet_json_congress(tweet_json: dict) -> dict:
    new_row = {}
    new_row["id_str"] = tweet_json["id"]

    dt = datetime.fromisoformat(tweet_json["date"])
    new_row["year"] = dt.year
    new_row["time_stamp"] = dt

    new_row["screen_name"] = tweet_json["screen_name"]
    new_row["text"] = tweet_json["text"]

    return new_row