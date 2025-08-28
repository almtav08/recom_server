from collections import defaultdict
from typing import List
from recommender.modules.hybridization.hibridization import Hybridization
from recommender.modules.recommendation.recommendation import Recommendation


class WeightedHybridization(Hybridization):
    def __init__(self, recommendations: List[Recommendation], weights: List[float], top: int = 10):
        super(Hybridization, self).__init__()
        self.recommendations = recommendations
        self.weights = weights
        self.top = top

    def hybridize(self):
        rec = defaultdict(lambda: {"score": 0, "sources": set()})

        # recolectamos scores y fuentes
        for idx, recom in enumerate(self.recommendations):
            source_name = type(recom).__name__
            for item, score in recom.get_recommendations():
                rec[item]["score"] += score * self.weights[idx]
                rec[item]["sources"].add(source_name)

        # generamos lista ordenada
        ranked = sorted(
            [(item, data["score"], data["sources"]) for item, data in rec.items()],
            key=lambda x: x[1],
            reverse=True
        )[:self.top]

        # añadimos explicaciones
        results = []
        for item, score, sources in ranked:
            if sources == {"ItemRecommendation"}:
                explanation = f"This resource is relevant based on your recent interactions."
            elif sources == {"UserRecommendation"}:
                explanation = f"This resource was studied by users similar to you."
            else:  # ambos
                explanation = f"This resource is important both for its characteristics and because similar users consulted it."

            results.append({
                "id": item,
                "score": score,
                "explanation": explanation
            })

        return results