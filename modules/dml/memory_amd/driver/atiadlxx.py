import ctypes as C
from modules.dml.memory_amd.driver.atiadlxx_apis import ADL2_Main_Control_Create, ADL_Main_Memory_Alloc, ADL2_Adapter_NumberOfAdapters_Get, ADL2_Adapter_AdapterInfo_Get, ADL2_Adapter_MemoryInfo2_Get, ADL2_Adapter_DedicatedVRAMUsage_Get, ADL2_Adapter_VRAMUsage_Get
from modules.dml.memory_amd.driver.atiadlxx_structures import ADL_CONTEXT_HANDLE, AdapterInfo, LPAdapterInfo, ADLMemoryInfo2
from modules.dml.memory_amd.driver.atiadlxx_defines import ADL_OK


class ATIADLxx:
    iHyperMemorySize = 0

    def __init__(self):
        self.context = ADL_CONTEXT_HANDLE()
        ADL2_Main_Control_Create(ADL_Main_Memory_Alloc, 1, C.byref(self.context))
        num_adapters = C.c_int(-1)
        ADL2_Adapter_NumberOfAdapters_Get(self.context, C.byref(num_adapters))
        AdapterInfoArray = (AdapterInfo * num_adapters.value)()
        ADL2_Adapter_AdapterInfo_Get(self.context, C.cast(AdapterInfoArray, LPAdapterInfo), C.sizeof(AdapterInfoArray))
        self.devices = []
        busNumbers = []
        for adapter in AdapterInfoArray:
            if adapter.iBusNumber not in busNumbers: # filter duplicate device
                self.devices.append(adapter)
                busNumbers.append(adapter.iBusNumber)
        self.iHyperMemorySize = self.get_memory_info2(0).iHyperMemorySize

    def get_memory_info2(self, adapterIndex: int) -> ADLMemoryInfo2:
        info = ADLMemoryInfo2()

        if ADL2_Adapter_MemoryInfo2_Get(self.context, adapterIndex, C.byref(info)) != ADL_OK:
            raise RuntimeError("ADL2: Failed to get MemoryInfo2")

        return info

    def get_dedicated_vram_usage(self, index: int) -> int:
        usage = C.c_int(-1)

        if ADL2_Adapter_DedicatedVRAMUsage_Get(self.context, self.devices[index].iAdapterIndex, C.byref(usage)) != ADL_OK:
            raise RuntimeError("ADL2: Failed to get DedicatedVRAMUsage")

        return usage.value

    def get_vram_usage(self, index: int) -> int:
        usage = C.c_int(-1)

        if ADL2_Adapter_VRAMUsage_Get(self.context, self.devices[index].iAdapterIndex, C.byref(usage)) != ADL_OK:
            raise RuntimeError("ADL2: Failed to get VRAMUsage")

        return usage.value
