import json
import torch
from tracerec.algorithms.graph_based.transe import TransE
from tracerec.data.triples.triples_manager import TriplesManager
from tracerec.samplers.path_based_sampler import PathBasedNegativeSampler


if __name__ == "__main__":
    # Create a sample triples manager with some triples
    with open(f'./moodle/data/prev_graph.json', 'r') as gd:
        prev_data: dict = json.load(gd)

    with open(f'./moodle/data/repeat_graph.json', 'r') as gb:
        back_data: dict = json.load(gb)

    triples = []
    relation_prev = 0
    relation_back = 1
    for head, tails in prev_data.items():
        for tail in tails:
            triples.append((int(head), relation_prev, int(tail)))

    for head, tails in back_data.items():
        for tail in tails:
            triples.append((int(head), relation_back, int(tail)))

    triples_manager = TriplesManager(triples)
    train_x, train_y, test_x, test_y = triples_manager.split(
        train_ratio=0.8, relation_ratio=True, random_state=42, device="cpu"
    )

    sampler = PathBasedNegativeSampler(
        triples_manager,
        corruption_ratio=0.5,
        device="cpu",
        min_distance=1.0,
    )
    train_x_neg = sampler.sample(train_x, num_samples=1, random_state=42)

    transe = TransE(
        num_entities=triples_manager.get_entity_count(),
        num_relations=2,
        embedding_dim=10,
        device="cpu",
        norm=1,
    )

    transe.to_device("cpu")
    transe.compile(
        optimizer=torch.optim.Adam, criterion=torch.nn.MarginRankingLoss(margin=1.0)
    )

    # Fit the model
    transe.fit(
        train_x,
        train_x_neg,
        num_epochs=1,
        batch_size=1,
        lr=0.001,
        verbose=True,
        checkpoint_path="./recommender/states/know.pth",
    )
