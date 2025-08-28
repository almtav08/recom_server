from asyncio import subprocess
import json
from typing import Dict, List, cast

import torch
from database.api_models import RecommendedResource
from database.async_query_gen import AsyncQueryGenerator
from database.models import User
from recommender.modules.hybridization.weighted import WeightedHybridization
from tracerec.algorithms.graph_based.transe import TransE
from tracerec.algorithms.graph_based.graph_embedder import GraphEmbedder
from tracerec.algorithms.sequential_based.sequential_embedder import SequentialEmbedder
from tracerec.data.paths.path_manager import PathManager
from recommender.modules.recommendation.item import ItemRecommendation
from recommender.modules.recommendation.user import UserRecommendation
from recommender.modules.recommender.recommender import Recommender
from utils.commons import calc_user_path


class HybridRecommender(Recommender):
    def __init__(self) -> None:
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kb_emb: GraphEmbedder = None
        self.cf_emb: SequentialEmbedder = None
        self.items: list = []
        self.users_emb: dict = None
        self.relations: list = []

    async def recommend(
        self, user: User, top: int, client: AsyncQueryGenerator
    ) -> List[RecommendedResource]:
        path = [
            item
            for item in user.resources
            if item.recid in list(self.items)
        ]

        interactions = await client.select_interactions_by_user_and_resource(user.id, path[-1].id)
        is_pass = interactions[-1].passed if len(path) > 0 else True

        # Collaborative filtering recommendations
        users = await client.select_users_by_resource_and_passed(path[-1].id, is_pass)
        if len(users) == 0:
            users = await client.select_users_by_pass()
        user_path = list(calc_user_path(user))
        path_manager = PathManager(paths={user.id: user_path}, max_seq_length=self.cf_emb.max_seq_length, item_embedder=self.kb_emb)
        user_path = path_manager.get_user_path(user.id)

        user_recom = UserRecommendation()\
            .calc_recommendations(self.users_emb, users, path[-1].recid, user, user_path, self.cf_emb, top)

        # Knowledge graph recommendations
        item_recom = ItemRecommendation()\
        .calc_recommendations(self.items, self.relations, is_pass, path[-1].recid, self.kb_emb, top)

        # Combine the recommendations
        hybridization = WeightedHybridization([user_recom, item_recom], [0.4, 0.6], top)

        recommendations = []

        for item in hybridization.hybridize():
            resource = await client.select_resource_by_recid(item['id'])
            recommend_resource = RecommendedResource(
                id=resource.id,
                name=resource.name,
                type=resource.type,
                reason=item['explanation']
            )
            recommendations.append(recommend_resource)
        return recommendations

    async def load(self, client: AsyncQueryGenerator):
        # Load the modules used for the recommendation
        self.kb_emb: TransE = torch.load("recommender/states/know.pth", weights_only=False)
        self.kb_emb.to(self.device)
        self.kb_emb.eval()
        self.kb_emb.requires_grad_(False)

        self.cf_emb = torch.load("recommender/states/path.pth", weights_only=False)
        self.cf_emb.to(self.device)
        self.cf_emb.eval()
        self.cf_emb.requires_grad_(False)

        items = await client.list_resources()
        self.items = list(map(lambda r: r.recid, items))
        self.relations = [0, 1] # 0 Prev, 1 Repeat

        self.users_emb = {}
        users = await client.select_users_by_pass()
        paths = {}
        for user in users:
            paths[user.id] = list(calc_user_path(user))

        path_manager = PathManager(paths=paths, max_seq_length=self.cf_emb.max_seq_length, item_embedder=self.kb_emb)
        for user in users:
            self.users_emb[user.id] = self.cf_emb(path_manager.get_user_path(user.id).unsqueeze(0)).detach()[0]

    def unload(self):
        self.kb_emb = None
        self.cf_emb = None
        self.users_emb = None
        self.back_graph = None

    def train(self):
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