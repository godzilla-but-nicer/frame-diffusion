import gzip
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce


with open("workflow/paths.json", "r") as path_file:
    paths = json.loads(path_file.read())

with open("workflow/config.json", "r") as config_file:
    config = json.loads(config_file.read())

pfdf = pd.read_csv("data/binary_frames/predicted_frames.tsv", sep="\t")


congress_frames_by_type = []
for frame_type in config["frames"].keys():
        if frame_type != "low_f1":
            congress_frames_by_type.append(pd.read_csv(f"data/binary_frames/congress/congress_{frame_type}.tsv", sep="\t"))

cfdf = reduce(lambda l, r: pd.merge(l, r.drop("text", axis="columns"),
              on="id_str"),
              congress_frames_by_type)

journalist_frames_by_type = []
for frame_type in config["frames"].keys():
        if frame_type != "low_f1":
            journalist_frames_by_type.append(pd.read_csv(f"data/binary_frames/journalists/journalists_{frame_type}.tsv", sep="\t"))

jfdf = reduce(lambda l, r: pd.merge(l, r.drop("text", axis="columns"),
              on="id_str"),
              congress_frames_by_type)

trump_frames_by_type = []
for frame_type in config["frames"].keys():
        if frame_type != "low_f1":
            trump_frames_by_type.append(pd.read_csv(f"data/binary_frames/trump/trump_{frame_type}.tsv", sep="\t"))

tfdf = reduce(lambda l, r: pd.merge(l, r.drop("text", axis="columns"),
              on="id_str"),
              congress_frames_by_type)


all_frames = pd.concat([pfdf, cfdf, jfdf, tfdf])
all_frames.to_csv("data/binary_frames/all_group_frames.tsv",
                  sep="\t",
                  index=False)
