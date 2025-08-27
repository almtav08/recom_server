import sys
import torch
from tracerec.algorithms.sequential_based.sasrec import SASRecEncoder
from tracerec.data.paths.path_manager import PathManager
from tracerec.losses.supcon import SupConLoss

sys.path.append("./")
from database.query_gen import QueryGenerator
from utils.commons import calc_user_path

if __name__ == "__main__":
    client = QueryGenerator()
    client.connect()
    users = client.select_users_by_pass()
    client.disconnect()

    paths = {}
    grades = []

    for idx, user in enumerate(users):
        paths[idx] = calc_user_path(user).tolist()
        grades.append(int(user.is_pass))

    graph_embedder = torch.load("./recommender/states/know.pth", weights_only=False)

    # Create a PathManager instance
    max_seq_length = 4
    path_manager = PathManager(paths, grades, max_seq_length, graph_embedder)

    train_x, train_y, train_masks, test_x, test_y, test_masks = path_manager.split(
        train_ratio=0.5, relation_ratio=True, random_state=42, device="cpu"
    )

    sasrec = SASRecEncoder(
        embedding_dim=10,
        max_seq_length=4,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
        device="cpu",
    )

    sasrec.compile(optimizer=torch.optim.Adam, criterion=SupConLoss())

    sasrec.fit(
        train_x,
        train_y,
        train_masks,
        num_epochs=1,
        batch_size=1,
        lr=0.001,
        verbose=True,
        checkpoint_path="./recommender/states/path.pth",
    )
