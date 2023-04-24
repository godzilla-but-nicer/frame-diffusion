from datetime import datetime


# we have to refer to these a bunch so lets put them in one place
immigration_keywords = "immigration|immigrants?|illegals|undocumented|illegal aliens?|migrants?|migration"

# these are the dates of the public tweets data
start_date = datetime.strptime("01-01-2018 00:00:00", "%m-%d-%Y %H:%M:%S")
end_date = datetime.strptime("12-31-2019 23:59:59", "%m-%d-%Y %H:%M:%S")

# all fields we will ever need
tweet_fields = ["id", "created_at", "text", "public_metrics", 
                "in_reply_to_user_id", "entities", "referenced_tweets"]
tweet_expansions = ["author_id", "referenced_tweets.id"]
user_fields = ["id", "name", "username"]