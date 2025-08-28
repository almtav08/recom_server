from collections import defaultdict
from typing import List
from database.async_query_gen import AsyncQueryGenerator
from database.models import Resource, User
from recommender.modules.recommendation.recommendation import Recommendation
from utils.commons import calc_user_path, check_pass_quiz, torch_cosine_similarity
from tracerec.algorithms.sequential_based.sequential_embedder import SequentialEmbedder


class UserRecommendation(Recommendation):
    def __init__(self):
        super(UserRecommendation, self).__init__()
        self.recommendations = []

    def calc_recommendations(self, user_embs, users: List[User], target_item: int, target_user: User, user_path, embedder: SequentialEmbedder, top: int):
        new_target_emb = embedder(user_path.unsqueeze(0)).detach()[0]
        scores = []
        for user in users:
            user_emb = user_embs.get(user.id)
            if user_emb is not None:
                score = torch_cosine_similarity(new_target_emb, user_emb)
                scores.append((user, score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:30]
        similar_users: List[User] = [user for user, _ in scores]

        recommendations_tmp = defaultdict(int)
        max_preference = 0
        for neig_user in similar_users:
            # Calculate the resource path for the neighbor user
            neig_path: List[int] = []
            for resource in neig_user.resources:
                if len(neig_path) == 0 or neig_path[-1] != resource.id:
                    neig_path.append(resource.id)
            # Obtain the ocurrence of the last item of the target user, if the item is not found, set the starting # index to 0
            last_item_idx = neig_path.index(target_item) + 1 if target_item in neig_path else 0
            # Iterate over the neighbor's path starting from the index after the last item
            for idx in range(last_item_idx, len(neig_path)):
                item_id = neig_path[idx]
                # The score is incremented by the inverse of the index + 1 to prioritize earlier items
                recommendations_tmp[item_id] += 1 / (idx + 1)
                # Update the maximum preference score if the current item's score is higher
                max_preference = max(max_preference, recommendations_tmp[item_id])
        # Normalize the recommendation scores by dividing by the maximum preference score
        for item_id in recommendations_tmp:
            recommendations_tmp[item_id] /= max_preference
        # If the item is already in the target user path it is removed
        for item in target_user.resources:
            recommendations_tmp.pop(item.id, None)
        recommendations_s = sorted(recommendations_tmp.items(), key=lambda x: x[1], reverse=True)[:top]
        self.recommendations = recommendations_s
        return self