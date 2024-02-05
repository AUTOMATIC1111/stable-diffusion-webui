from .driver.atiadlxx import ATIADLxx


class AMDMemoryProvider:
    driver: ATIADLxx = ATIADLxx()

    @staticmethod
    def mem_get_info(index):
        usage = AMDMemoryProvider.driver.get_dedicated_vram_usage(index) * (1 << 20)
        return (AMDMemoryProvider.driver.iHyperMemorySize - usage, AMDMemoryProvider.driver.iHyperMemorySize)
