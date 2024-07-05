"""
Generate a dataset of clusters based on the input configuration.

Example run:
python generate_clusters.py config/clusters_100_5.toml ~/data/algohertz/clustering/datasets

Copyright © 2024 AlgoHertz. License: MIT.
"""

import numpy as np
import pandas as pd
import random
import sys
import toml

RND_SEED = 42

random.seed(RND_SEED)
np.random.seed(RND_SEED)

toml_config_path = sys.argv[1]
dataset_dir_path = sys.argv[2]

def generate_cluster(size: int, dim_count: int, center_min: float, center_max: float, std_min: float, std_max: float):
    center = [random.uniform(center_min, center_max) for _ in range(dim_count)]
    std = [random.uniform(std_min, std_max) for _ in range(dim_count)]
    data = dict()
    for i in range(dim_count):
        values = np.random.normal(loc=center[i], scale=std[i], size=size)
        data[f"ax{i}"] = values
    df = pd.DataFrame.from_dict(data)
    return df

with open(toml_config_path, 'r') as file:
    config = toml.load(file)
    
clusters = []
for i in range(config["cluster_count"]):
    size = random.randint(config["min_cluster_size"], config["max_cluster_size"])
    df = generate_cluster(
        size,
        config["dim_count"],
        config["center_min"],
        config["center_max"],
        config["std_min"],
        config["std_max"],        
    )
    df["cluster"] = i
    clusters.append(df)

cluster_dataset = pd.concat(clusters)

dataset_out_path = dataset_dir_path + "/" + config["name"] + ".parquet"

cluster_dataset.to_parquet(dataset_out_path)
