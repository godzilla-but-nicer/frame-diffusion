# %%
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.linalg as sla
import tqdm
from simpletransformers.classification import MultiLabelClassificationModel as MLCM

from functools import reduce


with open("data/edge_lists/full_dyads.json") as json_file:
    dyads = json.loads(json_file.read())

with open("workflow/config.json") as config_file:
    config = json.loads(config_file.read())

print(len(dyads))

# %%
general = MLCM(model_type=config["model"],
               model_name=config["model_names"]["generic"],
               use_cuda=False)
specific = MLCM(model_type=config["model"],
                model_name=config["model_names"]["specific"],
                use_cuda=False)
narrative = MLCM(model_type=config["model"],
                 model_name=config["model_names"]["narrative"],
                 use_cuda=False)

models = {}
models["generic"] = general
models["specific"] = specific
models["narrative"] = narrative

# %%
# %%
framed_objects = []
for i, dyad in enumerate(tqdm.tqdm(dyads)):
    frame_dyad = {}
    obj = json.loads(dyad)
    if obj["relationship"] == "retweet":
        # classify the target only
        target_generic_frames = general.predict([obj["target_full"]["text"]])[0]
        target_specific_frames = specific.predict([obj["target_full"]["text"]])[0]
        target_narrative_frames = narrative.predict([obj["target_full"]["text"]])[0]

        target_frames = np.hstack((target_generic_frames,
                                   target_specific_frames,
                                   target_narrative_frames))

        frame_dyad["source_frames"] = target_frames
        frame_dyad["target_frames"] = target_frames

        frame_dyad["relationship"] = obj["relationship"]

        frame_dyad["source"] = obj["source_full"]["screen_name"]
        frame_dyad["target"] = obj["target_full"]["screen_name"]

        frame_dyad["created_at"] = obj["source_full"]["created_at"]

        framed_objects.append(frame_dyad)
    elif obj["relationship"] == "quote" or obj["relationship"] == "reply":
        target_generic_frames = general.predict([obj["target_full"]["text"]])[0]
        target_specific_frames = specific.predict([obj["target_full"]["text"]])[0]
        target_narrative_frames = narrative.predict([obj["target_full"]["text"]])[0]
        target_frames = np.hstack((target_generic_frames,
                                   target_specific_frames,
                                   target_narrative_frames))


        source_generic_frames = general.predict([obj["source_full"]["text"]])[0]
        source_specific_frames = specific.predict([obj["source_full"]["text"]])[0]
        source_narrative_frames = narrative.predict([obj["source_full"]["text"]])[0]
        source_frames = np.hstack((source_generic_frames,
                                   source_specific_frames,
                                   source_narrative_frames))


        frame_dyad["source_frames"] = source_frames
        frame_dyad["target_frames"] = target_frames

        frame_dyad["relationship"] = obj["relationship"]

        frame_dyad["source"] = obj["source_full"]["screen_name"]
        frame_dyad["target"] = obj["target_full"]["screen_name"]

        frame_dyad["created_at"] = obj["source_full"]["created_at"]
        
        framed_objects.append(frame_dyad)

with open("data/edge_lists/classified_dyads.json", "w") as fout:
    json.dump(framed_objects, fout)
