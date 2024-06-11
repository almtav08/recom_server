import torch
import torch.nn as nn
import torch.nn.functional as F
from recommender.module import RecommenderModule

class UserEmbedding(RecommenderModule):
    def __init__(self, hidden_dim, num_entities, embedding_dim, device, similarity_type):
        super(UserEmbedding, self).__init__()
        self.device = device
        # self.embedding = nn.Embedding(num_entities, embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim).to(device)
        self.linear = nn.Linear(hidden_dim, embedding_dim).to(device)
        self.similarity_type = similarity_type

    def forward(self, path: torch.Tensor):
        # embeds = self.embedding(path)
        lstm_out, _ = self.lstm(path.view(len(path), 1, -1))
        # output = self.linear(lstm_out.view(len(path), -1))
        output = self.linear(lstm_out[:, -1])
        output = torch.mean(output, dim=0, keepdim=True)
        output_scores = F.log_softmax(output, dim=1)
        return output_scores
    
    def embed(self, path: torch.Tensor) -> torch.Tensor:
        # embeds = self.embedding(path)
        lstm_out, _ = self.lstm(path.view(len(path), 1, -1))
        output = self.linear(lstm_out[:, -1])
        output = torch.mean(output, dim=0, keepdim=True)
        return output
    
    def get_simtype(self):
        return self.similarity_type