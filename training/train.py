import os
import sys
from typing import Dict, List
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import json
from pick import pick
import numpy as np
from db import QueryGenerator, Resource, User
from training.trainer import Trainer
from recommender import TransMethod, UserEmbedding, TransE, TransEWithTags, TransR, TransRWithTags, TransH, TransHWithTags, RotatE, RotatEWithTags, RecommenderModule
from utils import path_distance, calc_user_path, ContrastiveLoss, UserDatasetClusters, EntityDataset
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score

def load_kb_trainer(resources: List[Resource], device: torch.device, model_name: str):
    # Load Knowledge data
    with open(f'./moodle/data/prev_graph.json', 'r') as gd:
        graph_data: dict = json.load(gd)

    with open(f'./moodle/data/repeat_graph.json', 'r') as gb:
        graph_back: dict = json.load(gb)

    resource_dict: Dict[int, Resource] = {resource.id: resource for resource in resources}
    relations = {'is_previous_of': 0, 'needs_repeat': 1}
    triples = []
    for k, v in graph_data.items():
        for i in v:
            triples.append((resource_dict[int(k)].recid, relations['is_previous_of'], resource_dict[int(i)].recid))
    for k, v in graph_back.items():
        for i in v:
            triples.append((resource_dict[int(k)].recid, relations['needs_repeat'], resource_dict[int(i)].recid))

    # Load tags
    res_tags = [resource.type for resource in resource_dict.values()]
    unique_tags = list(set(res_tags))
    id_tags = {tag: i for i, tag in enumerate(unique_tags)}
    tags = {resource.recid: id_tags[resource.type] for resource in resource_dict.values()} 

    # Define module parameters
    num_entities = len(resources)
    num_relations = len(relations)
    embedding_dim = 10
    num_tags = len(unique_tags)
    project_dim = 10

    # Define training parameters
    num_epochs = 200
    kfolds = 10
    learning_rate = 0.01
    margin = 1.0
    min_distance = 3

    dataset = EntityDataset(triples, len(tags), min_distance, tags, device)

    add_project = f'{", project_dim" if "TransR" in model_name else ""}'
    add_tags = f'{", num_tags" if "WithTags" in model_name else ""}'
    model: TransMethod = eval(f"{model_name}(num_entities, num_relations, embedding_dim, device{add_project + add_tags})")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=margin, reduction='mean')

    return dataset, kfolds, num_epochs, model, optimizer, criterion

def load_cf_trainer(users: List[User], resources: List[Resource], device: torch.device, model_name: str):
    # Define module parameters
    hidden_dim = 40 # Second dimension of the LSTM Layer
    embedding_dim = 10 # Dimensionality of the embeddings
    num_entities = len(resources) # Total number of unique elements in the array

    # Define training parameters
    num_epochs = 1000
    kfolds = 10
    learning_rate = 0.1
    margin = 1.0
    num_clusters = 2

    # Embed paths
    kb_model: TransMethod = torch.load("states/TransE.pth")
    kb_model.eval()
    kb_model.requires_grad_(False)

    # Cluster users
    print("Assigning clusters to users...")
    paths = []
    user_paths = {}
    for user in users:
        path = calc_user_path(user)
        paths.append(path)
        user_paths[user.id] = path

    pairwise_distances = np.zeros((len(paths), len(paths)))
    for i, path1 in enumerate(paths):
        for j, path2 in enumerate(paths):
            if i != j:
                pairwise_distances[i, j] = path_distance(path1, path2, num_entities, device)
    pairwise_distances_condensed = squareform(pairwise_distances)

    linkage_method = 'complete'  # Choose linkage method (e.g., complete, single, average)
    cluster_tree = hierarchy.linkage(pairwise_distances_condensed, method=linkage_method)
    cluster_assignments = hierarchy.fcluster(cluster_tree, num_clusters, criterion='maxclust')
    silhouette_avg = silhouette_score(pairwise_distances, cluster_assignments)
    cluster_assignments = (cluster_assignments - 1).astype(np.int64)  # Adjust cluster assignments to start from 0
    print(f"Clusters assigned with a score of {silhouette_avg}")

    dataset = UserDatasetClusters(user_paths, cluster_assignments, kb_model, device)

    model = UserEmbedding(hidden_dim, num_entities, embedding_dim, device, model_name)
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(margin)

    return dataset, kfolds, num_epochs, model, optimizer, criterion

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        title = "Please provide the family model to train"
        options = ["Knowledge", "Collaborative"]
        model_family, _ = pick(options, title, indicator='=>')
    else:
        model_family = args[0]

    if len(args) < 2:
        title = "Please provide the model to train"
        options = ["TransE", "TransEWithTags", "TransR", "TransRWithTags", "TransH", "TransHWithTags", "RotatE", "RotatEWithTags"] if model_family == "Knowledge" else ["ContrastiveLoss"]
        model_name, _ = pick(options, title, indicator='=>')
    else:
        model_name = args[1]

    print(f"Training {model_name} model")

    # Load the items and users
    client = QueryGenerator()
    client.connect()
    resources = client.list_resources()
    users = client.list_users()
    client.disconnect()

    # Select the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    dataset, kfolds, num_epochs, model, optimizer, criterion = load_kb_trainer(resources, device, model_name) if model_family == "Knowledge" else load_cf_trainer(users, resources, device, model_name)

    # Train the module
    trainer = Trainer(dataset)
    trainer.kfold_train(kfolds, num_epochs, model, optimizer, criterion)