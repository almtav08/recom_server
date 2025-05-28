from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List
from .recommendation import Recommendation

class Hybridization(ABC):
    
    @abstractmethod
    def hybridize(self):
        pass

class WeightedHybridization(Hybridization):
    def __init__(self, recommendations: List[Recommendation], weights: List[float], top: int):
        super(Hybridization, self).__init__()
        self.recommendations = recommendations
        self.weights = weights
        self.top = top

    def hybridize(self):
        recommendations = defaultdict(int)
        for idx, recom in enumerate(self.recommendations):
            for item, score in recom.get_recommendations():
                recommendations[item] += score * self.weights[idx]

        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:self.top]

    