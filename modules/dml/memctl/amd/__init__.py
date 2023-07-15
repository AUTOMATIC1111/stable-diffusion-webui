from modules.dml.memctl.memctl import MemoryControl
from .driver.atiadlxx import ATIADLxx

class AMDMemoryControl(MemoryControl):
    driver: ATIADLxx = ATIADLxx()
    def mem_get_info(index):
        usage = AMDMemoryControl.driver.get_dedicated_vram_usage(index) * (1 << 20)
        return (AMDMemoryControl.driver.iHyperMemorySize - usage, AMDMemoryControl.driver.iHyperMemorySize)
