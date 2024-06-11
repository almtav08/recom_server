import torch
import torch.nn as nn
from recommender.module import RecommenderModule

class TransMethod(RecommenderModule):
    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(TransMethod, self).__init__()
        # Initialize the attributes
        self.device = device
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim).to(self.device)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim).to(self.device)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, heads, relations, tails, head_tags=None, tail_tags=None):
        raise NotImplementedError
    
    def embed(self, entity) -> torch.Tensor:
        return self.entity_embeddings(entity)

class TransE(TransMethod):
    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(TransE, self).__init__(num_entities, num_relations, embedding_dim, device)

    def forward(self, heads, relations, tails, head_tags=None, tail_tags=None):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        # Calculate the translation scores
        scores = head_embeddings + relation_embeddings - tail_embeddings
        return torch.linalg.vector_norm(scores, dim=-1)
    
class TransEWithTags(TransMethod):
    def __init__(self, num_entities, num_relations, embedding_dim, device, num_tags):
        super(TransEWithTags, self).__init__(num_entities, num_relations, embedding_dim, device)
        # Initialize the attributes
        self.num_tags = num_tags

        # Embeddings
        self.tag_embeddings = nn.Embedding(num_tags, embedding_dim).to(self.device)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.tag_embeddings.weight.data)

    def forward(self, heads, relations, tails, head_tags, tail_tags):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        head_tag_embeddings = self.tag_embeddings(head_tags)
        tail_tag_embeddings = self.tag_embeddings(tail_tags)

        # Calculate the translation scores
        scores = head_embeddings + relation_embeddings - tail_embeddings + head_tag_embeddings - tail_tag_embeddings
        return torch.linalg.vector_norm(scores, dim=-1)
    
class TransH(TransMethod):
    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(TransH, self).__init__(num_entities, num_relations, embedding_dim, device)
        # Embeddings
        self.normal_vectors = nn.Embedding(num_relations, embedding_dim).to(self.device)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.normal_vectors.weight.data)

    def forward(self, heads, relations, tails, head_tags=None, tail_tags=None):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        normal_vectors = self.normal_vectors(relations)

        # Project the head and tail embeddings onto the hyperplane
        head_proj = head_embeddings - torch.sum(head_embeddings * normal_vectors, dim=-1, keepdim=True) * normal_vectors
        tail_proj = tail_embeddings - torch.sum(tail_embeddings * normal_vectors, dim=-1, keepdim=True) * normal_vectors

        # Calculate the translation scores
        scores = head_proj + relation_embeddings - tail_proj
        return torch.linalg.vector_norm(scores, dim=-1)
    
class TransHWithTags(TransMethod):
    def __init__(self, num_entities, num_relations, embedding_dim, device, num_tags):
        super(TransHWithTags, self).__init__(num_entities, num_relations, embedding_dim, device)
        # Initialize the attributes
        self.num_tags = num_tags

        # Embeddings
        self.tag_embeddings = nn.Embedding(num_tags, embedding_dim).to(self.device)
        self.normal_vectors = nn.Embedding(num_relations, embedding_dim).to(self.device)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.tag_embeddings.weight.data)
        nn.init.xavier_uniform_(self.normal_vectors.weight.data)

    def forward(self, heads, relations, tails, head_tags, tail_tags):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        head_tag_embeddings = self.tag_embeddings(head_tags)
        tail_tag_embeddings = self.tag_embeddings(tail_tags)
        normal_vectors = self.normal_vectors(relations)

        # Project the head and tail embeddings onto the hyperplane
        head_proj = head_embeddings - torch.sum(head_embeddings * normal_vectors, dim=-1, keepdim=True) * normal_vectors
        tail_proj = tail_embeddings - torch.sum(tail_embeddings * normal_vectors, dim=-1, keepdim=True) * normal_vectors

        # Calculate the translation scores
        scores = head_proj + relation_embeddings - tail_proj + head_tag_embeddings - tail_tag_embeddings
        return torch.linalg.vector_norm(scores, dim=-1)
    
class TransR(TransMethod):
    def __init__(self, num_entities, num_relations, embedding_dim, device, projection_dim):
        super(TransR, self).__init__(num_entities, num_relations, embedding_dim, device)
        # Initialize the attributes
        self.projection_dim = projection_dim

        # Embeddings
        self.relation_projections = nn.Embedding(num_relations, embedding_dim * projection_dim).to(self.device)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.relation_projections.weight.data)

    def forward(self, heads, relations, tails, head_tags=None, tail_tags=None):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        relation_projections = self.relation_projections(relations).view(-1, self.embedding_dim, self.projection_dim)

        # Project the head and tail embeddings
        head_proj = torch.matmul(head_embeddings.unsqueeze(1), relation_projections).squeeze(1)
        tail_proj = torch.matmul(tail_embeddings.unsqueeze(1), relation_projections).squeeze(1)

        # Calculate the translation scores
        scores = head_proj + relation_embeddings - tail_proj
        return torch.linalg.vector_norm(scores, dim=-1)
    
class TransRWithTags(TransMethod):
    def __init__(self, num_entities, num_relations, embedding_dim, device, projection_dim, num_tags):
        super(TransRWithTags, self).__init__(num_entities, num_relations, embedding_dim, device)
        # Initialize the attributes
        self.num_tags = num_tags
        self.projection_dim = projection_dim

        # Embeddings
        self.tag_embeddings = nn.Embedding(num_tags, embedding_dim).to(self.device)
        self.relation_projections = nn.Embedding(num_relations, embedding_dim * projection_dim).to(self.device)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.tag_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_projections.weight.data)

    def forward(self, heads, relations, tails, head_tags, tail_tags):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        head_tag_embeddings = self.tag_embeddings(head_tags)
        tail_tag_embeddings = self.tag_embeddings(tail_tags)
        relation_projections = self.relation_projections(relations).view(-1, self.embedding_dim, self.projection_dim)

        # Project the head and tail embeddings
        head_proj = torch.matmul(head_embeddings.unsqueeze(1), relation_projections).squeeze(1)
        tail_proj = torch.matmul(tail_embeddings.unsqueeze(1), relation_projections).squeeze(1)
        head_tag_proj = torch.matmul(head_tag_embeddings.unsqueeze(1), relation_projections).squeeze(1)
        tail_tag_proj = torch.matmul(tail_tag_embeddings.unsqueeze(1), relation_projections).squeeze(1)

        # Calculate the translation scores
        scores = head_proj + relation_embeddings - tail_proj + head_tag_proj - tail_tag_proj
        return torch.linalg.vector_norm(scores, dim=-1)
    
class RotatE(TransMethod):
    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(RotatE, self).__init__(num_entities, num_relations, embedding_dim, device)

    def forward(self, heads, relations, tails, head_tags=None, tail_tags=None):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        # Calculate the translation scores
        re_head, im_head = torch.chunk(head_embeddings, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_embeddings, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation_embeddings, 2, dim=-1)

        re_relation = torch.cos(re_relation)
        im_relation = torch.sin(im_relation)

        re_head, im_head = re_head * re_relation - im_head * im_relation, re_head * im_relation + im_head * re_relation
        re_tail, im_tail = re_tail * re_relation - im_tail * im_relation, re_tail * im_relation + im_tail * re_relation
        scores = re_head * re_tail + im_head * im_tail
        return torch.linalg.vector_norm(scores, dim=-1)
    
class RotatEWithTags(TransMethod):
    def __init__(self, num_entities, num_relations, embedding_dim, device, num_tags):
        super(RotatEWithTags, self).__init__(num_entities, num_relations, embedding_dim, device)
        # Initialize the attributes
        self.num_tags = num_tags

        # Embeddings
        self.tag_embeddings = nn.Embedding(num_tags, embedding_dim).to(self.device)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.tag_embeddings.weight.data)

    def forward(self, heads, relations, tails, head_tags, tail_tags):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        head_tag_embeddings = self.tag_embeddings(head_tags)
        tail_tag_embeddings = self.tag_embeddings(tail_tags)

        # Calculate the translation scores
        re_head, im_head = torch.chunk(head_embeddings, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_embeddings, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation_embeddings, 2, dim=-1)
        re_head_tag, im_head_tag = torch.chunk(head_tag_embeddings, 2, dim=-1)
        re_tail_tag, im_tail_tag = torch.chunk(tail_tag_embeddings, 2, dim=-1)

        re_relation = torch.cos(re_relation)
        im_relation = torch.sin(im_relation)
        re_head, im_head = re_head * re_relation - im_head * im_relation, re_head * im_relation + im_head * re_relation
        re_tail, im_tail = re_tail * re_relation - im_tail * im_relation, re_tail * im_relation + im_tail * im_relation
        re_head_tag, im_head_tag = re_head_tag * re_relation - im_head_tag * im_relation, re_head_tag * im_relation + im_head_tag * im_relation
        re_tail_tag, im_tail_tag = re_tail_tag * re_relation - im_tail_tag * im_relation, re_tail_tag * im_relation + im_tail_tag * im_relation
        scores = re_head * re_tail + im_head * im_tail + re_head_tag * re_tail_tag + im_head_tag * im_tail_tag
        return torch.linalg.vector_norm(scores, dim=-1)