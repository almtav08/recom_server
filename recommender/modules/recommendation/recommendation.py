from abc import ABC, abstractmethod
from typing import List, Tuple


class Recommendation(ABC):
    def __init__(self):
        self.recommendations = []

    @abstractmethod
    def calc_recommendations():
        pass

    def get_recommendations(self) -> List[Tuple[int, int]]:
        return self.recommendations