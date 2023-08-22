import gzip
import json
import re
import pandas as pd
from typing import Dict, Optional
from tqdm import tqdm


def check_connections(tweet_json: Dict, group: str, check_mentions=False) -> Optional[Dict]:
    successors = identify_successors(tweet_json, group, check_mentions)

    signifiers = ["mention_0", "reply_to", "quote_of", "retweet_of"]

    for signifier in signifiers:
        if signifier in successors:
            return successors

    return None


def check_sample_group(edge_info: pd.Series, catalog: Dict,
                       kinds=["reply", "quote", "retweet"]) -> Dict:
    
    matches = {}
    for kind in kinds:
        matches.update(check_group_kind_single(edge_info, catalog, kind))
    
    return matches


def check_group_kind_single(edge_info: pd.Series, catalog: Dict, kind: str) -> Dict:

    if kind == "reply":
        target_tweet_id = "reply_to"
        target_user_id = "reply_to_user"
        target_user_name = "reply_to_user_name"
    elif kind == "retweet":
        target_tweet_id = "retweet_of"
        target_user_id = "retweet_of_user"
        target_user_name = "retweet_of_user_name"
    elif kind == "quote":
        target_tweet_id = "quote_of"
        target_user_id = "quote_of_user"
        target_user_name = "quote_of_user_name"

    edge_info_keys = [target_tweet_id, target_user_id, target_user_name]
    catalog_keys = ["tweet_id", "user_id", "user_name"]

    new_cols = {"tweet_id": edge_info["tweet_id"]}

    # make sure we have the column in edge_info and that it is not None
    for edge_key, cata_key in zip(edge_info_keys, catalog_keys):
        if edge_key in edge_info.keys():
            if edge_info[edge_key]:

                new_cols[edge_key + "_in_sample"] = None

                # check the catalog
                if str(edge_info[edge_key]).lower() in str(catalog[cata_key].keys()):
                    new_cols[edge_key + "_in_sample"] = catalog[cata_key][edge_info[edge_key]]

    return new_cols


def build_sample_catalog(paths: Dict) -> Dict:

    # catalog all the tweet ids, user ids, and screen names in our data

    # first lets build the basic structure of the catalog
    kinds = ["tweet_id", "user_id", "user_name"]
    groups = ["public", "congress", "journalists", "trump"]
    catalog = {k: {} for k in kinds}


    # now let's fill all of the public information
    years = ["2018", "2019"]
    for year in years:
        print(f"Cataloging {year} Public Tweets")
        for t in tqdm(gzip.open(paths["public"][f"{year}_json"])):
            tweet_json = json.loads(t)
            catalog["tweet_id"][str(tweet_json["id"])] = f"public_{year}"
            catalog["user_id"][str(tweet_json["user"]["id"])] = f"public_{year}"
            catalog["user_name"][str(tweet_json["user"]["screen_name"]).lower()] = f"public_{year}"

    # same for congress
    with open(paths["congress"]["tweet_json"], "r") as congress_json:
        congress_tweets = json.loads(congress_json.read())

    print(f"Cataloging Congress Tweets")
    for tweet_json in tqdm(congress_tweets):
        catalog["tweet_id"][str(tweet_json["id"])] = f"congress"
        catalog["user_id"][str(tweet_json["user_id"])] = "congress"
        catalog["user_name"][str(tweet_json["screen_name"]).lower()] = "congress"

    # finally for journalists
    with open(paths["journalists"]["tweet_json"], "r") as journalists_json:
        journalists_tweets = json.loads(journalists_json.read())

    print(f"Cataloging Journalists Tweets")
    for tweet_json in tqdm(journalists_tweets):
        catalog["tweet_id"][str(tweet_json["id"])] = "journalists"
        catalog["user_id"][str(tweet_json["author_id"])] = "journalists"
        catalog["user_name"][str(tweet_json["screen_name"]).lower()] = "journalists"

    return catalog


def identify_successors(tweet_json: Dict, group: str, check_mentions=False) -> Dict:

    succs = {}

    if group == "public":

        succs["tweet_id"] = str(tweet_json["id"])
        succs["user_id"] = str(tweet_json["user"]["id"])
        succs["user_name"] = tweet_json["user"]["screen_name"]

        if check_mentions:
            stripped_of_rt = re.sub(r"RT @[a-zA-Z0-9]+\W", "", tweet_json["text"])
            mentions = re.findall(r"@([a-zA-Z0-9]+)", stripped_of_rt)
            for m, mention in enumerate(mentions):
                succs["mention_" + str(m)] = mention

        if tweet_json["in_reply_to_status_id"]:
            succs["reply_to"] = str(tweet_json["in_reply_to_status_id"])
            succs["reply_to_user"] = str(tweet_json["in_reply_to_user_id"])

        elif "quoted_status" in tweet_json:
            succs["quote_of"] = str(tweet_json["quoted_status"]["id"])
            succs["quote_of_user"] = str(
                tweet_json["quoted_status"]["user"]["id"])

        elif "retweeted_status" in tweet_json:
            succs["retweet_of"] = str(tweet_json["retweeted_status"]["id"])
            succs["retweet_of_user"] = str(
                tweet_json["retweeted_status"]["user"]["id"])
        

    elif group == "journalists":

        succs["tweet_id"] = str(tweet_json["id"])
        succs["user_id"] = str(tweet_json["author_id"])
        succs["user_name"] = tweet_json["screen_name"]

        if check_mentions:
            stripped_of_rt = re.sub(r"RT @[a-zA-Z0-9]+\W", "", tweet_json["text"])
            mentions = re.findall(r"@([a-zA-Z0-9]+)", stripped_of_rt)
            for m, mention in enumerate(mentions):
                succs["mention_" + str(m)] = mention

        if "referenced_tweets" in tweet_json:

            for rt in tweet_json["referenced_tweets"]:

                if rt["type"] == "retweeted":

                    succs["retweet_of"] = str(rt["id"])
                    try:
                        succs["retweet_of_user_name"] = re.findall(
                            r"RT @([a-zA-Z0-9_]+):", tweet_json["text"])[0]
                    except:
                        raise ValueError(
                            f"Username not found in text: {tweet_json['text']}")

                elif rt["type"] == "quoted":

                    succs["quote_of"] = str(rt["id"])

                    # lets try parsing the url entities to get quoted user
                    for url in tweet_json["entities"]["urls"]:
                        username = re.match(
                            r"twitter.com/([a-zA-Z0-9]+)/status/" + str(rt["id"]), url["expanded_url"])

                        if username:
                            succs["quote_of_user_name"] = username.group(0)

                elif rt["type"] == "replied_to":
                    succs["reply_to"] = str(rt["id"])
                    succs["reply_to_user"] = str(
                        tweet_json["in_reply_to_user_id"])

    elif group == "congress":

        # this gets inserted if we find out edges
        id_info = {"tweet_id": tweet_json["id"],
                   "user_id": tweet_json["user_id"],
                   "user_name": tweet_json["screen_name"]}

        if check_mentions:
            stripped_of_rt = re.sub(r"RT @[a-zA-Z0-9]+\W", "", tweet_json["text"])
            mentions = re.findall(r"@([a-zA-Z0-9]+)", stripped_of_rt)
            for m, mention in enumerate(mentions):
                succs["mention_" + str(m)] = mention
                succs.update(id_info)

        if re.match(r"RT @[a-zA-Z0-9_]+", tweet_json["text"]):
            succs.update(id_info)
            try:
                succs["retweet_of_user_name"] = re.findall(
                    r"RT @([a-zA-Z0-9_]+)", tweet_json["text"])[0]
            except:
                raise ValueError(
                    f"Username not found in text: {tweet_json['text']}")

    else:
        raise ValueError(
            "`group` must be one of ['public', 'journalists', 'congress]")

    return succs
