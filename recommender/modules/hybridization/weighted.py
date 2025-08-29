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
        # Accumulate per-source scores without applying weights yet.
        # We'll apply weights only to items that appear in more than one source.
        rec = defaultdict(lambda: {"per_scores": defaultdict(float), "sources": set()})

        # recolectamos scores y fuentes por fuente (index)
        for idx, recom in enumerate(self.recommendations):
            source_name = type(recom).__name__
            for item, score in recom.get_recommendations():
                rec[item]["per_scores"][idx] += score
                rec[item]["sources"].add(source_name)

        # Sort by score descending so highest similarity/score comes first
        # Compute final score: if item present in multiple sources, apply weights;
        # otherwise keep the raw score (no weighting).
        aggregated = []
        for item, data in rec.items():
            per_scores = data["per_scores"]
            sources = data["sources"]
            if len(per_scores) > 1:
                final_score = sum(per_scores[i] * self.weights[i] for i in per_scores)
            else:
                # single source -> don't apply weights
                final_score = sum(per_scores.values())
            aggregated.append((item, final_score, sources))

        ranked = sorted(aggregated, key=lambda x: x[1], reverse=True)[:self.top]

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