import gzip
import json
import numpy as np
import pandas as pd
import tweet_handler as th

from glob import glob
from tqdm import tqdm

configfile: "workflow/config.json"

journalists = pd.read_csv("data/user_info/top_536_journos.tsv", sep="\t")["username"]


# the following rule gets all of the tweet data into the files we use for 
# classification and analysis it downloads the publically available 
# data and expects to find our data in particular places.
# Specifically these are:
#
#   - data/down_sample/edge_lists
#   - data/trump_twitter_archive.json
#   - data/immigration_tweets/US_2018.gz
#   - data/immigration_tweets/US_2019.gz
#   - data/decahose_retweets/*.gz   (tons of files)
#   - data/non_immigration_tweets/congress.json (gathered from a script but which one?)
#
# This rule should invoke the following rules:
#
#   - download_congress_tweets
#   - download_journalist_tweets
#   - filter_trump_immigration
#   - build_congress_table
#   - build_congress_nonimmigration_table
#   - build_trump_table
#   - build_journalist_table
#   - build_retweet_table
#   - collect_retweet_json
rule download_and_parse_tweets:
    input:
        "data/immigration_tweets/congress.json",
        expand("data/immigration_tweets/journalists/{journalist}.json", journalist=journalists),
        "data/immigration_tweets/trump.json",
        "data/immigration_tweets/public_sample.tsv",
        "data/non_immigration_tweets/congress.tsv",
        "data/immigration_tweets/trump.tsv",
        "data/immigration_tweets/journalists.tsv",
        expand("data/decahose_retweets/{year}_retweets.tsv", year=["2018", "2019"]),
        expand("data/decahose_retweets/{year}_retweets.gz", year=["2018", "2019"])


# 1. Download/extract the immigration tweet json data
# no api request
rule download_congress_tweets:
    input:
        "scripts/download_congress_immigration_tweets.py"
    output:
        "data/immigration_tweets/congress.json"
    script:
        "scripts/download_congress_immigration_tweets.py"


# takes multiple days, thousands of api requests
# elon destroyed this step of the workflow
rule download_journalist_tweets:
    input:
        "data/user_info/top_536_journos.tsv"
    output:
        expand("data/immigration_tweets/journalists/{journalist}.json", journalist=journalists)
    script:
        "scripts/download_journalist_tweets.py"


# pulling out the immigration subset from trump twitter archive
rule filter_trump_immigration:
    input:
        "data/trump_twitter_archive.json"
    output:
        "data/immigration_tweets/trump.json"
    script:
        "scripts/filter_trump_immigration_tweets.py"


# Unused?
rule get_us_tweet_ids_times:
    input:
        "data/immigration_tweets/US_2018.gz",
        "data/immigration_tweets/US_2019.gz"
    output:
        "data/us_public_ids.tsv"
    run:
        ids = []
        time_stamps = []
        screen_names = []
        for i in range(2):
            with gzip.open(input[i], 'r') as f:
                for i, line in enumerate(f):
                    tweet = th.load_tweet_obj(line)
                    ids.append(tweet["id_str"])
                    time_stamps.append(tweet["created_at"])
                    screen_names.append(tweet["user"]["screen_name"])

        
        df = pd.DataFrame({"id_str": ids, "time_stamp": time_stamps, "screen_name": screen_names})
        df.to_csv(output[0], sep="\t", index=False)


# 2. convert the json to tsv for each main category we're introducing
# unused?
rule build_public_sample_table:
    input:
        "data/immigration_tweets/US.gz",
        "data/us_public_ids.tsv"
    output:
        "data/immigration_tweets/public_sample.tsv"
    script:
        "scripts/take_public_sample.py"


rule build_congress_table:
    input:
        "data/immigration_tweets/congress.json"
    output:
        "data/immigration_tweets/congress.tsv"
    run:
        with open(input[0], "r") as json_file:
            tweets = json.loads(json_file.read())

        df_rows = []
        for tweet in tweets:
            row = th.parse_tweet_json(tweet, "congress")
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule build_congress_nonimmigration_table:
    input:
        "data/non_immigration_tweets/congress.json"
    output:
        "data/non_immigration_tweets/congress.tsv"
    run:
        with open(input[0], "r") as json_file:
            tweets = json.loads(json_file.read())

        df_rows = []
        for tweet in tweets:
            if "id" not in tweet:
                continue
            row = th.parse_tweet_json(tweet, "congress")
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule build_trump_table:
    input:
        "data/immigration_tweets/trump.json"
    output:
        "data/immigration_tweets/trump.tsv"
    run:
        with open(input[0], "r") as json_file:
            tweets = json.loads(json_file.read())
        
        df_rows = []
        for tweet in tweets:
            row = th.parse_tweet_json(tweet, "trump")
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule build_journalist_table:
    input:
        expand("data/immigration_tweets/journalists/{journalist}.json", journalist=journalists)
    output:
        "data/immigration_tweets/journalists.tsv"
    run:
        tweets = []
        for inp in input:
            with open(inp, "r") as json_file:
                tweets.extend(json.loads(json_file.read()))

        df_rows = []
        for tweet in tweets:
            row = th.parse_tweet_json(tweet, "v2")
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)

rule build_retweet_table:
    input:
        "data/decahose_retweets/"
    output:
        "data/decahose_retweets/{year}_retweets.tsv"
    run:
        input_files = glob(input[0] + f"/decahose.{wildcards.year}*.gz")

        all_df_rows = []
        for f in input_files:
            
            try:
                new_df_rows = []
                for json_str in gzip.open(f):
                    tweet = json.loads(json_str)
                    row = th.parse_tweet_json(tweet, "v1_retweet")
                    new_df_rows.append(row)
                
                all_df_rows.extend(new_df_rows)
            
            except:
                continue
            
        
        df = pd.DataFrame(all_df_rows)
        with open(output[0], "w") as tsv_file:
            df.to_csv(tsv_file, sep="\t", index=False)


rule collect_retweet_json:
    input:
        "data/decahose_retweets/"
    output:
        "data/decahose_retweets/{year}_retweets.gz"
    run:
        gzip_files = glob(f"data/decahose_retweets/decahose.{wildcards.year}-*.gz")

        json_entries = []
        for file in tqdm(gzip_files):
            try:
                for tweet in gzip.open(file):
                    part_json = json.loads(tweet)
                    json_entries.extend(part_json)

            except:
                print(f"File may be corrupted: {file}")
                continue


        with gzip.open(f"data/decahose_retweets/{wildcards.year}_retweets.gz", "wt", encoding="UTF-8") as gz_fout:
            json.dump(json_entries, gz_fout)


# The following rule gathers all of the classification rules into one
# It must be run after lots of things are in the right place which is
# possible after running the previous rules or the download_and_parse_tweets 
# collected rule.
# 
# Running this rule will invoke the following rules:
#
#   - classify_congress_tweets
#   - classify_trump_tweets
#   - classify_journalist_tweets
#   - classify_retweets
#   - combine_all_frames
rule classify_all_tweets:
    input:
        "data/binary_frames/congress/congress_generic.tsv",
        "data/binary_frames/congress/congress_specific.tsv",
        "data/binary_frames/congress/congress_narrative.tsv",
        "data/binary_frames/trump/trump_generic.tsv",
        "data/binary_frames/trump/trump_specific.tsv",
        "data/binary_frames/trump/trump_narrative.tsv",
        "data/binary_frames/journalists/journalists_generic.tsv",
        "data/binary_frames/journalists/journalists_specific.tsv",
        "data/binary_frames/journalists/journalists_narrative.tsv",
        "data/binary_frames/retweets/retweets_generic.tsv",
        "data/binary_frames/retweets/retweets_specific.tsv",
        "data/binary_frames/retweets/retweets_narrative.tsv",
        "data/binary_frames/all_group_frames.tsv"

# 3. Use classifier to identify frames in our tweet data
rule classify_congress_tweets:
    input:
        "data/immigration_tweets/congress.tsv"
    output:
        "data/binary_frames/congress/congress_generic.tsv",
        "data/binary_frames/congress/congress_specific.tsv",
        "data/binary_frames/congress/congress_narrative.tsv"
    shell:
        "python scripts/classify_tweet_frames.py congress"


rule classify_trump_tweets:
    input:
        "data/immigration_tweets/trump.tsv"
    output:
        "data/binary_frames/trump/trump_generic.tsv",
        "data/binary_frames/trump/trump_specific.tsv",
        "data/binary_frames/trump/trump_narrative.tsv"
    shell:
        "python scripts/classify_tweet_frames.py trump"


rule classify_journalist_tweets:
    input:
        "data/immigration_tweets/journalists.tsv"
    output:
        "data/binary_frames/journalists/journalists_generic.tsv",
        "data/binary_frames/journalists/journalists_specific.tsv",
        "data/binary_frames/journalists/journalists_narrative.tsv"
    shell:
        "python scripts/classify_tweet_frames.py journalists"


rule classify_retweets:
    input:
        expand("data/decahose_retweets/{year}_retweets.tsv", year=["2018", "2019"])
    output:
        "data/binary_frames/retweets/retweets_generic.tsv",
        "data/binary_frames/retweets/retweets_specific.tsv",
        "data/binary_frames/retweets/retweets_narrative.tsv"
    shell:
        "python scripts/classify_tweet_frames.py retweets"

rule combine_all_frames:
    input:
        expand("data/binary_frames/{group}/{group}_{frame_type}.tsv",
               group=["journalists", "congress", "trump", "retweets"],
               frame_type=["generic", "specific", "narrative"])
    output:
        "data/binary_frames/all_group_frames.tsv"

    shell:
        "python scripts/scripts/build_all_frame_df.py"

# Logistic regression and related tasks

# the regression itself, uses files built in rules below. probably want to rewrite
# the script as an actual script at some point
rule run_logistic_regession:
    input:
        "data/binary_frames/all_frames.tsv",
        "data/regression/features.tsv",
        "data/edge_lists/mention_neighbors_names.json"
    output:
        "data/regression/result_pickles/self_influence.pkl",
        "data/regression/result_pickles/alter_influence.pkl",
        "data/regression/self_influence_pairs.pkl",
        "data/regresion/alter_influence_pairs.pkl"
    shell:
        "python notebooks/logistic_regression.py"


# this actually builds both the features dataframe and the hashed mention network
# figure we might as well do it all in one pass over the frame df rather than
# break it out into two rules. it is a bit nastier though
rule build_features_df_for_regression:
    input:
        "data/binary_frames/all_frames.tsv",
        "data/user_info/user_id_map.tsv",
        "data/user_info/full_datasheet.tsv",
        "data/edge_lists/in_sample_mentions.tsv"
    output:
        "data/edge_lists/mention_neighbors.json",
        "data/edge_lists/mention_neighbors_names.json",
        "data/regression/features.tsv"
    shell:
        "python scripts/build_features_df_for_regression.py"


rule build_user_time_series_hash:
    input:
        "data/binary_frames/all_tweets.tsv"
    output:
        "data/binary_frames/user_time_series.pkl"
    shell:
        "python scripts/build_time_series_hash.py"


# Rules for all of the nondyad analysis
rule identify_successors:
    input:
        "data/immigration_tweets/US_2018.gz",
        "data/immigration_tweets/US_2019.gz",
        "data/immigration_tweets/journalists.json",
        "data/immigration_tweets/congress.json",
        "data/decahose_retweets/"
    output:
        "data/edge_lists/public_2018_successors.tsv",
        "data/edge_lists/public_2019_successors.tsv",
        "data/edge_lists/journalists_successors.tsv",
        "data/edge_lists/congress_successors.tsv",
        "data/edge_lists/retweet_successors.tsv"

    shell:
        "python scripts/extract_all_out_edges.py"

