import pandas as pd

journos = pd.read_csv("data/top_10k_journos.tsv", sep="\t")

top_536 = journos[journos["rank"] <= 536]

top_536.to_csv("data/top_536_journos.tsv", sep="\t", index=False)
