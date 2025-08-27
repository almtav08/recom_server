from abc import ABC, abstractmethod


class Recommender(ABC):
    
    @abstractmethod
    def recommend(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def train(self):
        pass