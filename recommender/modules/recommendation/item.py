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

        self.recommendations = sorted(scores, key=lambda x: x[1], reverse=True)[:top]
        return self

    # def __init__(self, embeddings: dict, top: int, back_graph):
    #     super(Recommendation, self).__init__()
    #     self.embeddings = embeddings
    #     self.top = top
    #     self.back_graph = back_graph
    #     self.recommendations = []
# 
    # def calc_recommendations(self, path: List[Resource], is_pass: bool, target_item):
    #     selected_items = self._get_selected_entities(path, is_pass)
    #     items_similarity = {item: torch_cosine_similarity(target_item, self.embeddings[item]) for item in # selected_items}
    #     recommendations = sorted(items_similarity.items(), key=lambda x: x[1], reverse=True)[:self.top]
# 
    #     self.recommendations = recommendations
    #     return self
# 
    # def _get_selected_entities(self, path: List[Resource], is_pass: bool):
    #     # Apply rules por selecting items
    #     resources = []
    #     if not is_pass:
    #         resources = [resource for resource in self.back_graph[path[-1].id]]
    #     if is_pass or len(resources) == 0:
    #         path_ids = [item.id for item in path]
    #         resources = [item for item in self.embeddings.keys() if item not in path_ids]
# 
    #     return resources