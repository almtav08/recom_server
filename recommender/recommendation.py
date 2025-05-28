from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple
from db import User, AsyncQueryGenerator, Resource
from utils import torch_cosine_similarity
from utils import check_pass_quiz

class Recommendation(ABC):

    @abstractmethod
    def calc_recommendations():
        pass

    @abstractmethod
    def get_recommendations() -> List[Tuple[int, int]]:
        pass

class ItemRecommendation(Recommendation):
    def __init__(self, embeddings: dict, top: int, back_graph):
        super(Recommendation, self).__init__()
        self.embeddings = embeddings
        self.top = top
        self.back_graph = back_graph
        self.recommendations = []

    def calc_recommendations(self, path: List[Resource], is_pass: bool, target_item):
        selected_items = self.get_selected_entities(path, is_pass)
        items_similarity = {item: torch_cosine_similarity(target_item, self.embeddings[item]) for item in selected_items}
        recommendations = sorted(items_similarity.items(), key=lambda x: x[1], reverse=True)[:self.top]

        self.recommendations = recommendations
        return self

    def get_recommendations(self):
        return self.recommendations

    def get_selected_entities(self, path: List[Resource], is_pass: bool):
        # Apply rules por selecting items
        resources = []
        if not is_pass:
            resources = [resource for resource in self.back_graph[path[-1].id]]
        if is_pass or len(resources) == 0:
            path_ids = [item.id for item in path]
            resources = [item for item in self.embeddings.keys() if item not in path_ids]

        return resources

class UserRecommendation(Recommendation):
    def __init__(self, embeddings: dict, user: User, top: int):
        super(Recommendation, self).__init__()
        self.embeddings = embeddings
        self.user = user
        self.top = top

    async def calc_recommendations(self, client: AsyncQueryGenerator, path: List[Resource], is_pass: bool, target_user):
        target_item = path[-1]
        selected_users = await self.get_selected_users(self.user, target_item, is_pass, client)
        user_similarity = {user: torch_cosine_similarity(self.embeddings[user.id], target_user) for user in selected_users}
        similar_users = sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)[:30]
        recommendations = defaultdict(int)

        # Initialize a variable to keep track of the maximum preference score
        max_preference = 0
        for neig_user, _ in similar_users:
            # Calculate the resource path for the neighbor user
            neig_path: List[int] = []
            for resource in neig_user.resources:
                if len(neig_path) == 0 or neig_path[-1] != resource.id:
                    neig_path.append(resource.id)
            # Obtain the ocurrence of the last item of the target user, if the item is not found, set the starting index to 0
            last_item_idx = neig_path.index(target_item.id) + 1 if target_item.id in neig_path else 0
            # Iterate over the neighbor's path starting from the index after the last item
            for idx in range(last_item_idx, len(neig_path)):
                item_id = neig_path[idx]
                # The score is incremented by the inverse of the index + 1 to prioritize earlier items
                recommendations[item_id] += 1 / (idx + 1)
                # Update the maximum preference score if the current item's score is higher
                max_preference = max(max_preference, recommendations[item_id])
        # Normalize the recommendation scores by dividing by the maximum preference score
        for item_id in recommendations:
            recommendations[item_id] /= max_preference
        # If the item is already in the target user path it is removed
        for item in path:
            recommendations.pop(item.id, None)
        recommendations_s = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:self.top]

        self.recommendations = recommendations_s
        return self

    def get_recommendations(self):
        return self.recommendations

    async def get_selected_users(self, user: User, resource: Resource, is_pass: bool, client: AsyncQueryGenerator) -> List[User]:
        # Apply rules for selecting users
        users = await client.list_users()
        users.remove(user)
        if not is_pass:
            users_r: List[User] = []
            for user_o in users:
                is_pass, _ = await check_pass_quiz(user_o.id, resource.quizid)
                if not is_pass:
                    users_r.append(user_o)
            return users_r
        return users