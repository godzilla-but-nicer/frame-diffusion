import gzip
import json
import re
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
