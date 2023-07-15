from modules.dml.memctl.memctl import MemoryControl

class UnknownMemoryControl(MemoryControl):
    def mem_get_info(index: int):
        return (1073741824, 1073741824)
