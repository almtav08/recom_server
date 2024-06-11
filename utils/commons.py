from typing import List
import numpy as np
import torch
import torch.nn as nn
from db import User

def torch_cosine_similarity(embedding1, embedding2):
    dot_product = torch.dot(embedding1, embedding2)
    norm_embedding1 = torch.linalg.norm(embedding1)
    norm_embedding2 = torch.linalg.norm(embedding2)
    similarity = dot_product / (norm_embedding1 * norm_embedding2)
    return similarity.cpu().numpy()[()]

def load_model_state(model: nn.Module, state_dict_path):
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def calc_user_path(user: User) -> List[int]:
    path = []
    for resource in user.resources:
        if len(path) == 0 or path[-1] != resource.recid:
            path.append(resource.recid)
    return path

def path_distance(path1, path2, dim, device):
    # Calculate the Laplacian matrix of the paths
    adj_path1 = torch.zeros((dim, dim), device=device)
    deg_path1 = torch.zeros((dim, dim), device=device)
    adj_path2 = torch.zeros((dim, dim), device=device)
    deg_path2 = torch.zeros((dim, dim), device=device)

    for i in range(1, len(path1)):
        adj_path1[path1[i-1], path1[i]] = 1
    deg_path1 = torch.diag(torch.sum(adj_path1, axis=0))

    for i in range(1, len(path2)):
        adj_path2[path2[i-1], path2[i]] = 1
    deg_path2 = torch.diag(torch.sum(adj_path2, axis=0))

    lap_path1 = deg_path1 - adj_path1
    lap_path2 = deg_path2 - adj_path2

    return torch.linalg.norm(lap_path1 - lap_path2, ord='fro') # Frobenius norm