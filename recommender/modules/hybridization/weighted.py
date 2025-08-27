from collections import defaultdict
from typing import List
from recommender.modules.hybridization.hibridization import Hybridization
from recommender.modules.recommendation.recommendation import Recommendation


class WeightedHybridization(Hybridization):
    def __init__(self, recommendations: List[Recommendation], weights: List[float], top: int):
        super(Hybridization, self).__init__()
        self.recommendations = recommendations
        self.weights = weights
        self.top = top

    def hybridize(self):
        rec = defaultdict(int)
        for idx, recom in enumerate(self.recommendations):
            for item, score in recom.get_recommendations():
                rec[item] += score * self.weights[idx]

        return sorted(rec.items(), key=lambda x: x[1], reverse=True)[:self.top]