from collections import defaultdict
from typing import Dict, List, Tuple
import torch
import pickle as pkl
from .module import RecommenderModule
from .recom import IRecommender
from db import AsyncQueryGenerator, User, Resource
from utils import torch_cosine_similarity, calc_user_path
from moodle import get_user_attempts, send_moodle_request, get_review_attempt

class HybridRecommender(IRecommender):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kb_model = None
        self.cf_model = None
        self.item_embeddings: dict = None
        self.users_embedding: dict = None
        self.back_graph: dict = None
        self.mid_to_qid: dict = None
        self.recid_to_mid: dict = None

    async def recommend(self, user: User, top: int, client: AsyncQueryGenerator) -> List[Resource]:
        path = [item for item in user.resources if item.id in list(self.item_embeddings.keys())]
        path_ids = [item.id for item in path]
        target_user_embedding = self.users_embeddings.get(user.id, self.cf_model.embed(self.kb_model.embed(torch.LongTensor(calc_user_path(user)).to(self.device))).detach()[0])
        last_item_embedding = self.item_embeddings[path[-1].id]
        is_pass, _ = await self.check_pass_quiz(user.id, path[-1].quizid) if path[-1].quizid else (True, [])

        # Collaborative filtering recommendations
        selected_users = await self.get_selected_users(user, path[-1], is_pass, client)
        user_similarity = {user: torch_cosine_similarity(self.users_embeddings[user.id], target_user_embedding) for user in selected_users}
        similar_users = sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)[:30]
        recommendations_cf = defaultdict(int)

        # Initialize a variable to keep track of the maximum preference score
        max_preference = 0
        for neig_user, _ in similar_users:
            # Calculate the resource path for the neighbor user
            neig_path: List[int] = []
            for resource in neig_user.resources:
                if len(neig_path) == 0 or neig_path[-1] != resource.id:
                    neig_path.append(resource.id)
            # Obtain the ocurrence of the last item of the target user, if the item is not found, set the starting index to 0
            last_item_idx = neig_path.index(path[-1].id) + 1 if path[-1].id in neig_path else 0
            # Iterate over the neighbor's path starting from the index after the last item
            for idx in range(last_item_idx, len(neig_path)):
                item_id = neig_path[idx]
                # The score is incremented by the inverse of the index + 1 to prioritize earlier items
                recommendations_cf[item_id] += 1 / (idx + 1)
                # Update the maximum preference score if the current item's score is higher
                max_preference = max(max_preference, recommendations_cf[item_id])
        # Normalize the recommendation scores by dividing by the maximum preference score
        for item_id in recommendations_cf:
            recommendations_cf[item_id] /= max_preference
        # If the item is already in the target user path it is removed
        for item in path_ids:
            recommendations_cf.pop(item, None)
        recommendations_cf_s = sorted(recommendations_cf.items(), key=lambda x: x[1], reverse=True)[:5]

        # Knowledge graph recommendations
        selected_items = self.get_selected_entities(path_ids, is_pass)
        items_similarity = {item: torch_cosine_similarity(last_item_embedding, self.item_embeddings[item]) for item in selected_items}
        recommendations_kb_s = sorted(items_similarity.items(), key=lambda x: x[1], reverse=True)[:5]

        # Combine the recommendations
        recommendations = defaultdict(int)
        for item, score in recommendations_cf_s:
            recommendations[item] += score * 0.4
        for item, score in recommendations_kb_s:
            recommendations[item] += score * 0.6

        recommendations_s = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top]
        
        recommendations = []
        for item, _ in recommendations_s:
            resource = await client.select_resource(item)
            recommendations.append(resource)
        return recommendations
    
    async def load(self, client: AsyncQueryGenerator):
        with open(f'./moodle/data/repeat_graph.pkl', 'rb') as bg:
            self.back_graph: Dict[int, List[int]] = pkl.load(bg)

        # Load the modules used for the recommendation
        self.kb_model: RecommenderModule = torch.load("states/TransE.pth").to(self.device)
        self.kb_model.eval()
        self.kb_model.requires_grad_(False)
        self.cf_model: RecommenderModule = torch.load("states/ContrastiveLoss.pth").to(self.device)
        self.cf_model.eval()
        self.cf_model.requires_grad_(False)

        # Load the embeddings
        self.item_embeddings = {}
        items = await client.list_resources()
        for item in items:
            self.item_embeddings[item.id] = self.kb_model.embed(torch.LongTensor([item.recid]).to(self.device)).detach()[0]

        self.users_embeddings = {}
        users = await client.list_users()
        for user in users:
            self.users_embeddings[user.id] = self.cf_model.embed(self.kb_model.embed(torch.LongTensor(calc_user_path(user)).to(self.device))).detach()[0]

    def unload(self):
        self.kb_model = None
        self.cf_model = None
        self.item_embeddings = None
        self.users_embeddings = None
        self.back_graph = None

    def get_selected_entities(self, path: List[int], is_pass: bool):
        resources = []
        if not is_pass:
            resources = [resource for resource in self.back_graph[path[-1]]]
        if is_pass or len(resources) == 0:
            resources = [item for item in self.item_embeddings.keys() if item not in path]

        return resources
    
    async def get_selected_users(self, user: User, resource: Resource, is_pass: bool, client: AsyncQueryGenerator) -> List[User]:
        users = await client.list_users()
        if not is_pass:
            users: List[User] = []
            for user in users:
                is_pass, _ = await self.check_pass_quiz(user.id, resource.quizid)[0]
                if not is_pass:
                    users.append(user)
        return users
    
    async def check_pass_quiz(self, user_id: int, quiz_id: int) -> Tuple[bool, List[bool]]:
        # Obtain last attempt data
        attempts = await send_moodle_request(get_user_attempts(user_id, quiz_id))
        last_attempt_id = int(attempts['attempts'][-1]['id'])

        # Check if the last attempt was successful
        result = await send_moodle_request(get_review_attempt(last_attempt_id))
        questions_grades = []
        maxmark, mark = 0, 0
        for question in result['questions']:
            maxmark += float(question['maxmark'])
            mark += float(question['mark'])
            questions_grades.append(mark == maxmark)
        return mark >= maxmark * 0.5
    
    def retrain(self):
        print("Retraining model")