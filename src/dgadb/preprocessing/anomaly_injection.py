import numpy as np
from src.dgadb.preprocessing import GraphDataContainer
import logging


class AnomalyInjection:
    def __init__(
        self,
        graph: GraphDataContainer,
        anom_set_id: int = 0,
        seed: int = 1234
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.seed = seed + anom_set_id

    def __check_contiguous_nodes(self):
        pass

    def inject(self):
        pass
