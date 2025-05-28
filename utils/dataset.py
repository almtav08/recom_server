import random
import networkx as nx
import torch
from torch.utils.data import Dataset
from recommender.module import RecommenderModule

class UserDataset(Dataset):
    def __init__(self, paths_dict):
        self.users = list(paths_dict.keys())
        self.paths_dict = paths_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        anchor_user = self.users[idx]  # Get anchor user
        anchor_path = self.paths_dict[anchor_user]  # Get anchor path
        
        # Sample a negative user (different from anchor user)
        negative_user = random.choice(self.users[:idx] + self.users[idx+1:])
        negative_path = self.paths_dict[negative_user]  # Get negative path
        return anchor_path, negative_path
    
class EntityDataset(Dataset):
    def __init__(self, triples, num_entities, min_distance, tags, device):
        self.triples = triples
        self.num_entities = num_entities
        self.min_distance = min_distance
        self.tags = tags
        self.device = device

        graph = nx.DiGraph()
        for head, rel, tail in triples:
            if rel == 0:
                graph.add_edge(head, tail)

        self.shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        # Sample the anchor triple
        anchor_triple = self.triples[idx]
        head, relation, tail = anchor_triple
        anchor_input = torch.LongTensor([head]).to(self.device), torch.LongTensor([relation]).to(self.device), torch.LongTensor([tail]).to(self.device), torch.LongTensor([self.tags[head]]).to(self.device), torch.LongTensor([self.tags[tail]]).to(self.device)

        # Sample a negative triple
        if relation == 0: # Is the previous relation
            candidates = [node for node, distance in self.shortest_paths[head].items() if distance >= self.min_distance]
        if relation == 1: # Is the repeat relation
            candidates = []
            for triple in self.triples:
                if triple[0] == tail and triple[1] == 0:
                    candidates.append(triple[2])
                if triple[2] == tail and triple[1] == 0:
                    candidates.append(triple[0])
    
        if not candidates:
            candidates = list(set(range(0, self.num_entities)) - {head} - {tail})

        negative_tail = random.choice(candidates)
        while (head, relation, negative_tail) in self.triples:
            candidates.remove(negative_tail)

            if not candidates:
                candidates = list(set(range(0, self.num_entities)) - {head} - {tail})

            negative_tail = random.choice(candidates)
        # negative_triple = (head, relation, negative_tail)
        negative_input = torch.LongTensor([head]).to(self.device), torch.LongTensor([relation]).to(self.device), torch.LongTensor([negative_tail]).to(self.device), torch.LongTensor([self.tags[head]]).to(self.device), torch.LongTensor([self.tags[negative_tail]]).to(self.device)

        return anchor_input, negative_input, torch.tensor([-1.0]).to(self.device)
    
class UserDatasetClusters(Dataset):
    def __init__(self, paths_dict: dict, clusters, kb_model: RecommenderModule, device):
        self.users = list(paths_dict.keys())
        self.paths_dict = paths_dict
        self.clusters = clusters
        self.kb_model = kb_model
        self.device = device

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # Sample the anchor user path
        anchor_user = self.users[idx]
        anchor_path = self.paths_dict[anchor_user]
        anchor_cluster = self.clusters[idx]
        
        # Sample a negative user path of a different cluster
        negative_idx = random.choice([i for i, c in enumerate(self.clusters) if c != anchor_cluster])
        negative_user = self.users[negative_idx]
        negative_path = self.paths_dict[negative_user]
        anchor_input = self.kb_model.embed(torch.LongTensor(anchor_path).to(self.device))
        negative_input = self.kb_model.embed(torch.LongTensor(negative_path).to(self.device))
        return anchor_input, negative_input, torch.tensor([-1.0]).to(self.device)