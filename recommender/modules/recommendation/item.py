import math
import torch
from recommender.modules.recommendation.recommendation import Recommendation


class ItemRecommendation(Recommendation):
    def __init__(self):
        super(ItemRecommendation, self).__init__()
        self.recommendations = []

    def calc_recommendations(
        self,
        entities,
        relations,
        is_pass,
        target,
        embedder,
        top,
        missing_entities=None,
        used_entities=None,
        max_mix_ratio=0.8,
    ):
        rel = relations[int(not is_pass)]
        scores = []
        for entity in entities:
            if target == entity or (is_pass and entity in used_entities):
                continue
            input_tensor = torch.tensor([[target, rel, entity]], dtype=torch.long)
            score = embedder(input_tensor)
            # Convert tensor/torch output to Python float for later normalization
            try:
                numeric = float(score.item()) if hasattr(score, "item") else float(score)
            except Exception:
                # fallback: cast via numpy
                numeric = float(score.detach().cpu().numpy()) if hasattr(score, "detach") else float(score)
            scores.append((entity, numeric))

        missing_scores = []
        if missing_entities and used_entities:
            for m_entity in missing_entities:
                if target == m_entity:
                    continue

                for u_entity in used_entities:
                    input_tensor = torch.tensor([[u_entity, rel, m_entity]], dtype=torch.long)
                    score = embedder(input_tensor)
                    # ensure numeric
                    try:
                        m_numeric = float(score.item()) if hasattr(score, "item") else float(score)
                    except Exception:
                        m_numeric = float(score.detach().cpu().numpy()) if hasattr(score, "detach") else float(score)
                    missing_scores.append((m_entity, m_numeric))

            best_missing = {}
            for entity, score in missing_scores:
                if entity not in best_missing or score < best_missing[entity]:
                    best_missing[entity] = score
            missing_scores = list(best_missing.items())

        # At this point the embedder returns lower==better. Convert raw scores
        # to similarity values in [0,1] where 1.0 is best.
        all_values = [s for _, s in scores] + [s for _, s in missing_scores]
        if len(all_values) == 0:
            self.recommendations = []
            return self

        min_s = min(all_values)
        max_s = max(all_values)

        def to_similarity(v):
            if max_s > min_s:
                norm = (v - min_s) / (max_s - min_s)
                return 1.0 - norm
            else:
                return 1.0

        scores = [(e, to_similarity(s)) for e, s in scores]
        missing_scores = [(e, to_similarity(s)) for e, s in missing_scores]

        # Sort so that highest similarity is first
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        missing_scores = sorted(missing_scores, key=lambda x: x[1], reverse=True)

        if missing_entities:
            mix_ratio = len(missing_entities) / (len(missing_entities) + len(entities))
            mix_ratio = min(mix_ratio, max_mix_ratio)  # limitamos a un máximo
        else:
            mix_ratio = 0.0

        n_missing = int(top * mix_ratio)
        n_regular = top - n_missing

        final_recommendations = missing_scores[:n_missing] + scores[:n_regular]

        self.recommendations = final_recommendations
        return self