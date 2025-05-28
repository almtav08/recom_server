import copy
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from recommender import RecommenderModule
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

class Trainer():
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def kfold_train(self, kfolds: int, epochs: int, base_model: RecommenderModule, optimizer: Optimizer, criterion: nn.Module):
        # Initialize the best loss
        best_loss = float('inf') 
        best_model_state_dict = None

        # Cross-validation loop
        kfold_loader = KFold(n_splits=kfolds, shuffle=True)
        for fold, (train_indices, val_indices) in enumerate(kfold_loader.split(self.dataset)):
            print(f"Fold {fold + 1}/{kfolds}")

            # Create data loaders for training and validation splits
            train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)
            train_dataloader, val_dataloader = DataLoader(self.dataset, sampler=train_sampler), DataLoader(self.dataset, sampler=val_sampler)

            # Prepare the model
            model = copy.deepcopy(base_model)

            # Training loop
            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                for anchor_input, negative_input, target in train_dataloader:
                    # Clear torch gradients
                    optimizer.zero_grad()
                    # Obtain the scores
                    scores_positive = model(*anchor_input)
                    print(scores_positive)
                    scores_negative = model(*negative_input)
                    print(scores_negative)
                    print(target)
                    exit()

                    # Compute the loss
                    loss = criterion(scores_positive, scores_negative, target)
                    total_loss += loss.item()

                    # Backpropagation
                    loss.backward()
                    optimizer.step()

                avg_loss = total_loss / len(train_indices)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

                # Validation loop
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for anchor_input, negative_input, target in val_dataloader:
                        # Obtain the scores
                        scores_positive = model(*anchor_input)
                        scores_negative = model(*negative_input)

                        # Compute the loss
                        loss = criterion(scores_positive, scores_negative, target)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_indices)
                print(f"Validation Loss: {avg_val_loss}")

                # Check if we have a new best model
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    best_model_state_dict = model.state_dict()

        # Save the best model
        if best_model_state_dict is not None:
            torch.save(model, f"states/{model.__class__.__name__}.pth")
