from modules.dml.optimizer.optimizer import Optimizer
from .driver.atiadlxx import ATIADLxx

class AMDOptimizer(Optimizer):
    driver: ATIADLxx = ATIADLxx()
    def memory_stats(index):
        return (AMDOptimizer.driver.iHyperMemorySize, AMDOptimizer.driver.get_dedicated_vram_usage(index))
