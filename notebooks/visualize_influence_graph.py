# %%
import numpy as np
import networkx as nx
import pandas as pd

sgc = pd.read_csv("../data/time_series_output/significant_complete_granger_partisan_True.tsv", index_col=0, sep="\t")

# %%
sgc_events_conditional = sgc[sgc["events_causing"] == True]
sgc_no_events = sgc[sgc["events_causing"] == False]
# %%
grouped_sgc_events = sgc_events_conditional.groupby(["source", "target"]).count()["frame"]
grouped_sgc_no_events = sgc_no_events.groupby(["source", "target"]).count()["frame"]
# %%
grouped_sgc_events = grouped_sgc_events.reset_index()
grouped_sgc_events["weight"] = grouped_sgc_events["frame"]
sgc_events_network = grouped_sgc_events.drop("frame", axis="columns")

grouped_sgc_no_events = grouped_sgc_no_events.reset_index()
grouped_sgc_no_events["weight"] = grouped_sgc_no_events["frame"]
sgc_network = grouped_sgc_no_events.drop("frame", axis="columns")
# %%
D_events = nx.from_pandas_edgelist(sgc_events_network, edge_attr="weight", create_using=nx.DiGraph)
D_users = nx.from_pandas_edgelist(sgc_network, edge_attr="weight", create_using=nx.DiGraph)
# %%
nx.write_graphml(D_events, "../data/time_series_output/influence_network_with_events.graphml")
nx.write_graphml(D_users, "../data/time_series_output/influence_network_without_events.graphml")
# %%
sgc_events = sgc[sgc["events_causing"] == True]
sgc_no_events = sgc[sgc["events_causing"] == False]


# %%
