import json
from typing import Dict

from edge_finder import identify_successors, check_connections, check_sample_group

def test_congress_retweet_mention():
    successors = identify_successors(congress_tweet(), "congress")
    print(successors)
    assert successors["user_id"] == "2853309155"
    assert successors["tweet_id"] == "948280895772950535"
    assert successors["mention_0"] == "RepCurbelo"
    assert successors["retweet_of_user_name"] == "FoxBusiness"


def test_journalist_reply():
    successors = identify_successors(journalist_tweet(), "journalists")
    assert successors["user_id"] == "38936142"
    assert successors["tweet_id"] == "1174838654168182784"
    assert successors["reply_to_user"] == "2853309155"
    assert successors["reply_to"] == "948280895772950535"


def test_public_quote():
    successors = identify_successors(public_tweet(), "public")
    print(successors)
    assert str(successors["user_id"]) == "3301798743"
    assert successors["tweet_id"] == "1053262168534278144"
    assert successors["quote_of"] == "1053110153883643905"
    assert successors["quote_of_user"] == "457984599"


def test_check_connections():
    connected = check_connections(congress_tweet(), "congress")
    disconnected = check_connections(disconnected_congress_tweet(), "congress")

    assert disconnected is None
    assert connected["mention_0"] == "RepCurbelo"


def test_check_sample_group():
    public_successors = identify_successors(public_tweet(), "public")
    congress_successors = identify_successors(congress_tweet(), "congress")
    journalist_successors = identify_successors(journalist_tweet(), "journalists")
    catalog = build_dummy_catalog()


    public_check = check_sample_group(public_successors, catalog)


    assert public_check["quote_of_in_sample"] is None

    journalist_check = check_sample_group(journalist_successors, catalog)

    print(journalist_check)

    assert journalist_check["reply_to_user_in_sample"] == "congress"
    assert journalist_check["reply_to_in_sample"] == "congress"

    congress_check = check_sample_group(congress_successors, catalog)
    print("congress successors", congress_successors)
    print("congress check", congress_check)

    assert congress_check["retweet_of_user_name_in_sample"] is None


def build_dummy_catalog() -> Dict:
    # sort of copying the code in the actual catalog function to do a halfassed test kind of hting
    kinds = ["tweet_id", "user_id", "user_name"]
    groups = ["public", "congress", "journalists", "trump"]
    catalog = {k: {} for k in kinds}


    # public tweet
    tweet_json = public_tweet()
    catalog["tweet_id"][str(tweet_json["id"])] = "public"
    catalog["user_id"][str(tweet_json["user"]["id"])] = "public"
    catalog["user_name"][str(tweet_json["user"]["screen_name"]).lower()] = "public"
    
    # congress tweet
    tweet_json = congress_tweet()
    catalog["tweet_id"][str(tweet_json["id"])] = "congress"
    catalog["user_id"][str(tweet_json["user_id"])] = "congress"
    catalog["user_name"][str(tweet_json["screen_name"]).lower()] = "congress"

    # journalist
    tweet_json = journalist_tweet()
    catalog["tweet_id"][str(tweet_json["id"])] = "journalists"
    catalog["user_id"][str(tweet_json["author_id"])] = "journalists"
    catalog["user_name"][str(tweet_json["screen_name"]).lower()] = "journalists"

    return catalog

# these are altered from the raw data so that they refer to one another for testing edge detection
def congress_tweet() -> Dict:
    json_string = """
    {
        "id": "948280895772950535",
        "screen_name": "RepCurbelo",
        "user_id": "2853309155",
        "time": "2018-01-02T14:52:38-05:00",
        "link": "https://www.twitter.com/FoxBusiness/statuses/948228024939446277",
        "text": "RT @FoxBusiness .@RepCurbelo: \\"This immigration issue is certainly front and center for all lawmakers now.\\" http://pbs.twimg.com/media/DSjIIa6XkAIfPyx.jpg https://video.twimg.com/amplify_video/948225448298901504/vid/320x180/3Y0V_irbGMVYhSeK.mp4",
        "source": "Twitter for iPhone"
    }
    """

    return json.loads(json_string)


def disconnected_congress_tweet() -> Dict:
    json_string = """
    {
        "id": "948280895772950535",
        "screen_name": "RepCurbelo",
        "user_id": "2853309155",
        "time": "2018-01-02T14:52:38-05:00",
        "link": "https://www.twitter.com/FoxBusiness/statuses/948228024939446277",
        "text": "\\"This immigration issue is certainly front and center for all lawmakers now.\\" http://pbs.twimg.com/media/DSjIIa6XkAIfPyx.jpg https://video.twimg.com/amplify_video/948225448298901504/vid/320x180/3Y0V_irbGMVYhSeK.mp4",
        "source": "Twitter for iPhone"
    }
    """

    return json.loads(json_string)


def journalist_tweet() -> Dict:
    json_string = """
    {
        "edit_history_tweet_ids": [
            "1174838654168182784"
        ],
        "public_metrics": {
            "retweet_count": 157,
            "reply_count": 10,
            "like_count": 309,
            "quote_count": 4,
            "impression_count": 0
        },
        "entities": {
            "urls": [
                {
                    "start": 151,
                    "end": 174,
                    "url": "https://t.co/2tJ7NfZmcm",
                    "expanded_url": "https://www.washingtonpost.com/immigration/trump-officials-considering-plan-to-divert-billions-of-dollars-in-additional-funds-for-border-barrier/2019/09/19/52897dce-d652-11e9-9610-fb56c5522e1c_story.html",
                    "display_url": "washingtonpost.com/immigration/tr\u2026",
                    "status": 200,
                    "unwound_url": "https://www.washingtonpost.com/immigration/trump-officials-considering-plan-to-divert-billions-of-dollars-in-additional-funds-for-border-barrier/2019/09/19/52897dce-d652-11e9-9610-fb56c5522e1c_story.html"
                }
            ],
            "annotations": [
                {
                    "start": 0,
                    "end": 8,
                    "probability": 0.6903,
                    "type": "Organization",
                    "normalized_text": "Democrats"
                },
                {
                    "start": 13,
                    "end": 16,
                    "probability": 0.2988,
                    "type": "Organization",
                    "normalized_text": "Hill"
                },
                {
                    "start": 34,
                    "end": 43,
                    "probability": 0.7741,
                    "type": "Organization",
                    "normalized_text": "Army Corps"
                },
                {
                    "start": 126,
                    "end": 136,
                    "probability": 0.6703,
                    "type": "Place",
                    "normalized_text": "White House"
                }
            ]
        },
        "author_id": "38936142",
        "referenced_tweets": [
            {
                "type": "replied_to",
                "id": "948280895772950535"
            }
        ],
        "text": "Democrats on Hill have called for Army Corps Chief to come give info, as they are now investigating contracting processes and White House involvement. https://t.co/2tJ7NfZmcm",
        "created_at": "2019-09-20T00:12:17.000Z",
        "id": "1174838654168182784",
        "in_reply_to_user_id": "2853309155",
        "screen_name": "jdawsey1"
    }
    """

    return json.loads(json_string)


def public_tweet() -> Dict:
    json_string = \
    """
    {
        "created_at": "Fri Oct 19 12:30:44 +0000 2018",
        "id": 1053262168534278144,
        "id_str": "1053262168534278144",
        "text": "People voting legally will not be intimidated by following laws. Lynch must support votes from  illegals and the de\u2026 https://t.co/MVMcIAZ6w2",
        "source": "<a href=\\"http://twitter.com\\" rel=\\"nofollow\\">Twitter Web Client</a>",
        "truncated": true,
        "in_reply_to_status_id": null,
        "in_reply_to_status_id_str": null,
        "in_reply_to_user_id": null,
        "in_reply_to_user_id_str": null,
        "in_reply_to_screen_name": null,
        "user": {
            "id": 3301798743,
            "id_str": "3301798743",
            "name": "Fran Butkiewicz",
            "screen_name": "gambler1647",
            "location": null,
            "url": null,
            "description": null,
            "translator_type": "none",
            "protected": false,
            "verified": false,
            "followers_count": 539,
            "friends_count": 690,
            "listed_count": 10,
            "favourites_count": 17624,
            "statuses_count": 36551,
            "created_at": "Thu May 28 15:05:23 +0000 2015",
            "utc_offset": null,
            "time_zone": null,
            "geo_enabled": false,
            "lang": "en",
            "contributors_enabled": false,
            "is_translator": false,
            "profile_background_color": "C0DEED",
            "profile_background_image_url": "http://abs.twimg.com/images/themes/theme1/bg.png",
            "profile_background_image_url_https": "https://abs.twimg.com/images/themes/theme1/bg.png",
            "profile_background_tile": false,
            "profile_link_color": "1DA1F2",
            "profile_sidebar_border_color": "C0DEED",
            "profile_sidebar_fill_color": "DDEEF6",
            "profile_text_color": "333333",
            "profile_use_background_image": true,
            "profile_image_url": "http://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png",
            "profile_image_url_https": "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png",
            "default_profile": true,
            "default_profile_image": false,
            "following": null,
            "follow_request_sent": null,
            "notifications": null
        },
        "geo": null,
        "coordinates": null,
        "place": null,
        "contributors": null,
        "quoted_status_id": 1053110153883643905,
        "quoted_status_id_str": "1053110153883643905",
        "quoted_status": {
            "created_at": "Fri Oct 19 02:26:41 +0000 2018",
            "id": 1053110153883643905,
            "id_str": "1053110153883643905",
            "text": "https://t.co/a1nWE4yUgS",
            "source": "<a href=\\"http://www.socialflow.com\\" rel=\\"nofollow\\">SocialFlow</a>",
            "truncated": false,
            "in_reply_to_status_id": null,
            "in_reply_to_status_id_str": null,
            "in_reply_to_user_id": null,
            "in_reply_to_user_id_str": null,
            "in_reply_to_screen_name": null,
            "user": {
                "id": 457984599,
                "id_str": "457984599",
                "name": "Breitbart News",
                "screen_name": "BreitbartNews",
                "location": null,
                "url": "http://breitbart.com",
                "description": "News, commentary, and destruction of the political/media establishment.",
                "translator_type": "none",
                "protected": false,
                "verified": true,
                "followers_count": 1009944,
                "friends_count": 102,
                "listed_count": 8525,
                "favourites_count": 959,
                "statuses_count": 97998,
                "created_at": "Sun Jan 08 01:50:52 +0000 2012",
                "utc_offset": null,
                "time_zone": null,
                "geo_enabled": false,
                "lang": "en",
                "contributors_enabled": false,
                "is_translator": false,
                "profile_background_color": "0C0D0D",
                "profile_background_image_url": "http://abs.twimg.com/images/themes/theme1/bg.png",
                "profile_background_image_url_https": "https://abs.twimg.com/images/themes/theme1/bg.png",
                "profile_background_tile": true,
                "profile_link_color": "FF5402",
                "profile_sidebar_border_color": "FFFFFF",
                "profile_sidebar_fill_color": "DDEEF6",
                "profile_text_color": "333333",
                "profile_use_background_image": true,
                "profile_image_url": "http://pbs.twimg.com/profile_images/949270171755077632/dw3M-58z_normal.jpg",
                "profile_image_url_https": "https://pbs.twimg.com/profile_images/949270171755077632/dw3M-58z_normal.jpg",
                "profile_banner_url": "https://pbs.twimg.com/profile_banners/457984599/1359997459",
                "default_profile": false,
                "default_profile_image": false,
                "following": null,
                "follow_request_sent": null,
                "notifications": null
            },
            "geo": null,
            "coordinates": null,
            "place": null,
            "contributors": null,
            "is_quote_status": false,
            "quote_count": 87,
            "reply_count": 575,
            "retweet_count": 224,
            "favorite_count": 308,
            "entities": {
                "hashtags": [],
                "urls": [
                    {
                        "url": "https://t.co/a1nWE4yUgS",
                        "expanded_url": "https://trib.al/epOGacP",
                        "display_url": "trib.al/epOGacP",
                        "unwound": {
                            "url": "https://www.breitbart.com/video/2018/10/18/lynch-voter-id-laws-early-voting-restrictions-designed-to-intimidate/",
                            "status": 200,
                            "title": "Lynch: Voter ID Laws, Early Voting Restrictions 'Designed To Intimidate' | Breitbart",
                            "description": "On Thursday's broadcast of MSNBC's \\"Hardball,\\" former Attorney General Loretta Lynch argued that restrictions on early voting and voter ID laws are | Video"
                        },
                        "indices": [
                            0,
                            23
                        ]
                    }
                ],
                "user_mentions": [],
                "symbols": []
            },
            "favorited": false,
            "retweeted": false,
            "possibly_sensitive": false,
            "filter_level": "low",
            "lang": "und"
        },
        "quoted_status_permalink": {
            "url": "https://t.co/DoRjAwWqGk",
            "expanded": "https://twitter.com/BreitbartNews/status/1053110153883643905",
            "display": "twitter.com/BreitbartNews/\u2026"
        },
        "is_quote_status": true,
        "extended_tweet": {
            "full_text": "People voting legally will not be intimidated by following laws. Lynch must support votes from  illegals and the dead or even those who vote in multiple districts. She assumes people are stupid---but decent, legal citizens know that the right to vote is important &amp; vote legally.",
            "display_text_range": [
                0,
                283
            ],
            "entities": {
                "hashtags": [],
                "urls": [],
                "user_mentions": [],
                "symbols": []
            }
        },
        "quote_count": 0,
        "reply_count": 0,
        "retweet_count": 0,
        "favorite_count": 0,
        "entities": {
            "hashtags": [],
            "urls": [
                {
                    "url": "https://t.co/MVMcIAZ6w2",
                    "expanded_url": "https://twitter.com/i/web/status/1053262168534278144",
                    "display_url": "twitter.com/i/web/status/1\u2026",
                    "indices": [
                        117,
                        140
                    ]
                }
            ],
            "user_mentions": [],
            "symbols": []
        },
        "favorited": false,
        "retweeted": false,
        "filter_level": "low",
        "lang": "en",
        "timestamp_ms": "1539952244300"
    }
    """

    return json.loads(json_string)