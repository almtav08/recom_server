from typing import Dict, List
import torch
import json
import subprocess
import os

from training.trainer import Trainer
from .hybridization import WeightedHybridization
from .recommendation import UserRecommendation, ItemRecommendation
from .module import RecommenderModule
from .recom import IRecommender
from db import AsyncQueryGenerator, User, Resource
from utils import calc_user_path, check_pass_quiz


class HybridRecommender(IRecommender):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kb_module = None
        self.cf_module = None
        self.item_embeddings: dict = None
        self.users_embeddings: dict = None
        self.back_graph: dict = None

    async def recommend(
        self, user: User, top: int, client: AsyncQueryGenerator
    ) -> List[Resource]:
        path = [
            item
            for item in user.resources
            if item.id in list(self.item_embeddings.keys())
        ]
        target_user_embedding = self.users_embeddings.get(
            user.id,
            self.cf_module.embed(
                self.kb_module.embed(
                    torch.LongTensor(calc_user_path(user)).to(self.device)
                )
            ).detach()[0],
        )
        last_item_embedding = self.item_embeddings[path[-1].id]
        is_pass, _ = (
            await check_pass_quiz(user.id, path[-1].quizid)
            if path[-1].quizid
            else (True, [])
        )

        # Collaborative filtering recommendations
        user_recom = await UserRecommendation(
            self.users_embeddings, user, top
        ).calc_recommendations(client, path, is_pass, target_user_embedding)

        # Knowledge graph recommendations
        item_recom = ItemRecommendation(
            self.item_embeddings, top, self.back_graph
        ).calc_recommendations(path, is_pass, last_item_embedding)

        # Combine the recommendations
        hybridization = WeightedHybridization([user_recom, item_recom], [0.4, 0.6], top)

        recommendations = []
        for item, _ in hybridization.hybridize():
            resource = await client.select_resource(item)
            recommendations.append(resource)
        return recommendations

    async def load(self, client: AsyncQueryGenerator):
        with open(f"./moodle/data/repeat_graph.json", "r") as bg:
            self.back_graph: Dict[int, List[int]] = json.load(bg)

        # Load the modules used for the recommendation
        self.kb_module: RecommenderModule = torch.load("states/TransE.pth").to(
            self.device
        )
        self.kb_module.eval()
        self.kb_module.requires_grad_(False)
        self.cf_module: RecommenderModule = torch.load("states/UserEmbedding.pth").to(
            self.device
        )
        self.cf_module.eval()
        self.cf_module.requires_grad_(False)

        # Load the embeddings
        self.item_embeddings = {}
        items = await client.list_resources()
        for item in items:
            self.item_embeddings[item.id] = self.kb_module.embed(
                torch.LongTensor([item.recid]).to(self.device)
            ).detach()[0]

        self.users_embeddings = {}
        users = await client.list_users()
        for user in users:
            self.users_embeddings[user.id] = self.cf_module.embed(
                self.kb_module.embed(
                    torch.LongTensor(calc_user_path(user)).to(self.device)
                )
            ).detach()[0]

    def unload(self):
        self.kb_module = None
        self.cf_module = None
        self.item_embeddings = None
        self.users_embeddings = None
        self.back_graph = None

    def retrain(self):
        # Entrenar el modelo Knowledge con TransE
        print("Entrenando modelo Knowledge (TransE)...")
        subprocess.run(
            ["python", "training/train.py", "Knowledge", "TransE"], check=True
        )

        # Entrenar el modelo Collaborative con ContrastiveLoss
        print("Entrenando modelo Collaborative (ContrastiveLoss)...")
        subprocess.run(
            ["python", "training/train.py", "Collaborative", "ContrastiveLoss"],
            check=True,
        )

        print("Entrenamiento completado. Los modelos han sido actualizados.")
