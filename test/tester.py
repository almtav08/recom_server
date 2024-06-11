import os
import sys
sys.path.append(os.getcwd())
import torch
import pickle as pkl
from typing import List
from pick import pick
from recommender import TransMethod, UserEmbedding
from utils import torch_cosine_similarity, calc_user_path
from db import QueryGenerator

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

    print(f"Testing {model_name} model")

    # Load the items and users
    client = QueryGenerator()
    client.connect()
    resources = client.list_resources()
    users = client.list_users()
    client.disconnect()

    # Select the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if model_family == "Knowledge":
        # Test the model
        kb_model: TransMethod = torch.load(f"states/{model_name}.pth").to(device)
        kb_model.eval()
        kb_model.requires_grad_(False)
        base_entity = kb_model.embed(torch.LongTensor([resources[0].recid]).to(device)).detach()[0]

        # Get top 5 similar entities with their names
        entity_similarity = {}
        for entity in resources[1:]:
            entity_embedding = kb_model.embed(torch.LongTensor([entity.recid]).to(device)).detach()[0]
            entity_similarity[entity.id] = torch_cosine_similarity(base_entity, entity_embedding)

        for entity, similarity in sorted(entity_similarity.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"Entity with id {entity}, Similarity: {similarity}")

    elif model_family == "Collaborative":
        cf_model: UserEmbedding = torch.load(f"states/{model_name}.pth").to(device)
        cf_model.eval()
        cf_model.requires_grad_(False)
        kb_model: TransMethod = torch.load(f"states/TransE.pth").to(device)
        kb_model.eval()
        kb_model.requires_grad_(False)

        base_user = users[2]
        path_input = torch.LongTensor(calc_user_path(base_user)).to(device)
        base_user_embedding = cf_model.embed(kb_model.embed(path_input)).detach()[0]

        user_similarity = {}
        for user in users:
            if user.id == base_user.id:
                continue
            path_input = torch.LongTensor(calc_user_path(user)).to(device)
            user_embedding = cf_model.embed(kb_model.embed(path_input)).detach()[0]
            user_similarity[user.id] = torch_cosine_similarity(base_user_embedding, user_embedding)

        for user, similarity in sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"User with id {user}: Similarity: {similarity}")

        print("------------")
         
        for user, similarity in sorted(user_similarity.items(), key=lambda x: x[1], reverse=False)[:5]:
            print(f"User with id {user}: Similarity: {similarity}")
        print("\n")