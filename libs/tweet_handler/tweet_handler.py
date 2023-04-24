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
                                        tweet_fields=tweet_fields,
                                        user_fields=user_fields,
                                        expansions=tweet_expansions,
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
