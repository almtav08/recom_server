import os
import sys
from typing import Dict, List
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
from pick import pick
import numpy as np
from db import QueryGenerator, Resource, User
from recommender import TransMethod, UserEmbedding, TransE, TransEWithTags, TransR, TransRWithTags, TransH, TransHWithTags, RotatE, RotatEWithTags
from utils import path_distance, calc_user_path, ContrastiveLoss, UserDatasetClusters, EntityDataset
from torch.utils.data import DataLoader
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score

def train_knowledge(graph_data: dict, graph_back: dict, resources: Dict[int, Resource], device: torch.device, model_name: str) -> None:
    # Create the triples and relations
    relations = {'is_previous_of': 0, 'needs_repeat': 1}
    triples = []
    for k, v in graph_data.items():
        for i in v:
            triples.append((resources[k].recid, relations['is_previous_of'], resources[i].recid))
    for k, v in graph_back.items():
        for i in v:
            triples.append((resources[k].recid, relations['needs_repeat'], resources[i].recid))

    # Load tags
    res_tags = [resource.type for resource in resources.values()]
    unique_tags = list(set(res_tags))
    id_tags = {tag: i for i, tag in enumerate(unique_tags)}
    tags = {resource.recid: id_tags[resource.type] for resource in resources.values()} 

    print(f"Loaded {len(triples)} triples and {len(tags)} tags")

    # Define module parameters
    num_entities = len(resources)
    num_relations = len(relations)
    embedding_dim = 10
    num_tags = len(unique_tags)
    project_dim = 10

    # Define training parameters
    num_epochs = 200
    learning_rate = 0.01
    margin = 1.0
    min_distance = 3

    add_project = f'{", project_dim" if "TransR" in model_name else ""}'
    add_tags = f'{", num_tags" if "WithTags" in model_name else ""}'
    model: TransMethod = eval(f"{model_name}(num_entities, num_relations, embedding_dim, device{add_project + add_tags})")

    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=margin, reduction='mean')

    # Initialize the best loss
    best_loss = float('inf') 
    best_model_state_dict = None

    # Laod the dataset
    dataset = EntityDataset(triples, len(tags), min_distance)
    dataloader = DataLoader(dataset, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for anchor_triple, negative_triple in dataloader:
            # Clear torch gradients
            optimizer.zero_grad()
            
            # Obtain positive scores
            head, relation, tail = anchor_triple
            heads, relations, tails = torch.LongTensor([head]).to(device), torch.LongTensor([relation]).to(device), torch.LongTensor([tail]).to(device)
            head_tags, tail_tags = torch.LongTensor([tags[head.item()]]).to(device), torch.LongTensor([tags[tail.item()]]).to(device)
            scores_positive = model(heads, relations, tails, head_tags, tail_tags)
            
            # Obtain negative scores
            n_head, n_relation, n_tail = negative_triple
            n_heads, n_relations, n_tails = torch.LongTensor([n_head]).to(device), torch.LongTensor([n_relation]).to(device), torch.LongTensor([n_tail]).to(device)
            n_head_tags, n_tail_tags = torch.LongTensor([tags[n_head.item()]]).to(device), torch.LongTensor([tags[n_tail.item()]]).to(device)
            scores_negative = model(n_heads, n_relations, n_tails, n_head_tags, n_tail_tags)

            # Compute the loss
            target = torch.tensor([-1.0]).to(device)
            loss = criterion(scores_positive, scores_negative, target)
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(triples)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state_dict = model.state_dict()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(triples)}")
    
    # Save the best model
    if best_model_state_dict is not None:
        torch.save(model, f"states/{model.__class__.__name__}.pth")

def train_collaborative(users: List[User], resources: Dict[int, Resource], device: torch.device, model_name: str) -> None:
    # Define module parameters
    hidden_dim = 40 # Second dimension of the LSTM Layer
    embedding_dim = 10 # Dimensionality of the embeddings
    num_entities = len(resources) # Total number of unique elements in the array

    # Define training parameters
    num_epochs = 1000
    learning_rate = 0.1
    margin = 1.0
    num_clusters = 2

    # Embed paths
    kb_model: TransMethod = torch.load("states/TransE.pth")
    kb_model.requires_grad_(False)

    # Initialize the UserEmbedding model
    model = UserEmbedding(hidden_dim, num_entities, embedding_dim, device, model_name)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    best_model_state_dict = None

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

    dataset = UserDatasetClusters(user_paths, cluster_assignments)
    dataloader = DataLoader(dataset, shuffle=True)

    loss_function = ContrastiveLoss(margin) if model.similarity_type == "ContrastiveLoss" else nn.NLLLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for anchor_path, negative_path in dataloader:
            # Clear torch gradients
            optimizer.zero_grad()

            # Prepare user data and compute loss
            anchor_inputs = torch.LongTensor(anchor_path).to(device)
            anchor_scores = model(kb_model.embed(anchor_inputs))

            if model.similarity_type == "ContrastiveLoss":
                negative_inputs = torch.LongTensor(negative_path).to(device)
                negative_scores = model(kb_model.embed(negative_inputs))
                target = torch.tensor([-1.0]).to(device)
                loss = loss_function(anchor_scores, negative_scores, target)

            elif model.similarity_type == "Clustering":
                target = torch.tensor([cluster_assignments[i]]).to(device)
                loss = loss_function(anchor_scores, target)

            # Update model
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(users)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state_dict = model.state_dict()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(users)}")

    if best_model_state_dict is not None:
        torch.save(model, f"states/{model.similarity_type}.pth")

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
        options = ["TransE", "TransEWithTags", "TransR", "TransRWithTags", "TransH", "TransHWithTags", "RotatE", "RotatEWithTags"] if model_family == "Knowledge" else ["ContrastiveLoss", "Clustering"]
        model_name, _ = pick(options, title, indicator='=>')
    else:
        model_name = args[1]

    print(f"Training {model_name} model")

    # Load the graph data
    with open(f'./moodle/data/prev_graph.pkl', 'rb') as gd:
        graph_data: dict = pkl.load(gd)

    with open(f'./moodle/data/repeat_graph.pkl', 'rb') as gb:
        graph_back: dict = pkl.load(gb)

    # Load the items and users
    client = QueryGenerator()
    client.connect()
    resources = client.list_resources()
    users = client.list_users()
    client.disconnect()

    resource_dict: Dict[int, Resource] = {resource.id: resource for resource in resources}

    # Select the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    train_knowledge(graph_data, graph_back, resource_dict, device, model_name) if model_family == "Knowledge" else train_collaborative(users, resource_dict, device, model_name)