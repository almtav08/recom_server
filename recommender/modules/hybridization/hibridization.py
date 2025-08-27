from abc import ABC, abstractmethod


class Hybridization(ABC):
    
    @abstractmethod
    def hybridize(self):
        pass