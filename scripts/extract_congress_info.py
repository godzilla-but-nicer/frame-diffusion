import json
import pandas as pd
from typing import List

with open("data/user_info/historical-users-filtered.json", "r") as fin:
    congress = json.loads(fin.read())

def flatten_congress_json(obj: dict) -> List[dict]:
    new_rows = []
    for account in obj["accounts"]:
        row = {}
        row["screen_name"] = account["screen_name"]
        row["name"] = obj["name"]
        row["type"] = obj["type"]
        row["chamber"] = obj["chamber"]        
        
        if "party" in obj:
            row["party"] = obj["party"]
        if "state" in obj:
            row["state"] = obj["state"]

        new_rows.append(row)
    
    return new_rows

rows = []
for item in congress:
    rows.extend(flatten_congress_json(item))

congress_info = pd.DataFrame(rows)
congress_info.to_csv("data/user_info/congress_info.tsv", sep="\t")
