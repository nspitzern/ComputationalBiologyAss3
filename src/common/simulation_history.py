from typing import List


class SimulationHistory:
    def __init__(self) -> None:
        self.__worst: List[float] = []
        self.__average: List[float] = []
        self.__best: List[float] = []

    @property
    def worst(self) -> List[float]:
        return self.__worst
    
    @property
    def average(self) -> List[float]:
        return self.__average
    
    @property
    def best(self) -> List[float]:
        return self.__best
    
    def add(self, worst, average, best) -> None:
        self.__worst.append(worst)
        self.__average.append(average)
        self.__best.append(best)
    
    def last_n_best_change(self, n: int) -> float:
        last_n = self.__best[-n:]
        return max(last_n) - min(last_n)

    def __len__(self):
        return len(self.__average)