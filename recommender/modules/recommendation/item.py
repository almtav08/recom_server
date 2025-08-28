import torch
from recommender.modules.recommendation.recommendation import Recommendation


class ItemRecommendation(Recommendation):
    def __init__(self):
        super(ItemRecommendation, self).__init__()
        self.recommendations = []

    def calc_recommendations(self, entities, relations, is_pass, target, embedder, top):
        rel = relations[int(not is_pass)]
        scores = []
        for entity in entities[:18]:
            if target == entity:
                continue
            input_tensor = torch.tensor([[target, rel, entity]], dtype=torch.long)
            score = embedder(input_tensor)
            scores.append((entity, score))

        self.recommendations = sorted(scores, key=lambda x: x[1])[:top]
        return self