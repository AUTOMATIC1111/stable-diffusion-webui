from abc import *
from typing import *

class Optimizer(metaclass=ABCMeta):
    driver: Any = None
    @abstractmethod
    def memory_stats(index: int) -> Tuple[int, int]:
        pass
