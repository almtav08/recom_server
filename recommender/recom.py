from abc import ABC, abstractmethod

class IRecommender(ABC):
    
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
    def retrain(self):
        pass