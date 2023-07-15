from abc import *
from typing import *

class MemoryControl(metaclass=ABCMeta):
    driver: Any = None
    @abstractmethod
    def mem_get_info(index: int) -> Tuple[int, int]:
        pass
