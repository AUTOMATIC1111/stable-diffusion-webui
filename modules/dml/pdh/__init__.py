from ctypes import *
from ctypes.wintypes import *
from typing import NamedTuple, TypeVar

from .apis import PdhExpandWildCardPathW, PdhOpenQueryW, PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetFormattedCounterValue, PdhGetFormattedCounterArrayW, PdhCloseQuery
from .structures import PDH_HQUERY, PDH_HCOUNTER, PDH_FMT_COUNTERVALUE, PPDH_FMT_COUNTERVALUE_ITEM_W
from .defines import *
from .msvcrt import malloc
from .errors import PDHError

class __InternalAbstraction(NamedTuple):
    flag: int
    attr_name: str

_type_map = {
    int: __InternalAbstraction(PDH_FMT_LARGE, "largeValue"),
    float: __InternalAbstraction(PDH_FMT_DOUBLE, "doubleValue"),
}

def expand_wildcard_path(path: str) -> list[str]:
    listLength = DWORD(0)
    if PdhExpandWildCardPathW(None, LPCWSTR(path), None, byref(listLength), PDH_NOEXPANDCOUNTERS) != PDH_MORE_DATA:
        raise PDHError("Something went wrong.")
    expanded = (WCHAR * listLength.value)()
    if PdhExpandWildCardPathW(None, LPCWSTR(path), expanded, byref(listLength), PDH_NOEXPANDCOUNTERS) != PDH_OK:
        raise PDHError(f"Couldn't expand wildcard path '{path}'")
    result = list()
    cur = str()
    for chr in expanded:
        if chr == '\0':
            result.append(cur)
            cur = str()
        else:
            cur += chr
    result.pop()
    return result

T = TypeVar("T", *_type_map.keys())

class HCounter(PDH_HCOUNTER):
    def get_formatted_value(self, type: T) -> T:
        if type not in _type_map:
            raise PDHError(f"Invalid value type: {type}")
        flag, attr_name = _type_map[type]
        value = PDH_FMT_COUNTERVALUE()
        if PdhGetFormattedCounterValue(self, DWORD(flag | PDH_FMT_NOSCALE), None, byref(value)) != PDH_OK:
            raise PDHError("Couldn't get formatted counter value.")
        return getattr(value.u, attr_name)

    def get_formatted_dict(self, type: T) -> dict[str, T]:
        if type not in _type_map:
            raise PDHError(f"Invalid value type: {type}")
        flag, attr_name = _type_map[type]
        bufferSize = DWORD(0)
        itemCount = DWORD(0)
        if PdhGetFormattedCounterArrayW(self, DWORD(flag | PDH_FMT_NOSCALE), byref(bufferSize), byref(itemCount), None) != PDH_MORE_DATA:
            raise PDHError("Something went wrong.")
        itemBuffer = cast(malloc(c_size_t(bufferSize.value)), PPDH_FMT_COUNTERVALUE_ITEM_W)
        if PdhGetFormattedCounterArrayW(self, DWORD(flag | PDH_FMT_NOSCALE), byref(bufferSize), byref(itemCount), itemBuffer) != PDH_OK:
            raise PDHError("Couldn't get formatted counter array.")
        result: dict[str, T] = dict()
        for i in range(0, itemCount.value):
            item = itemBuffer[i]
            result[item.szName] = getattr(item.FmtValue.u, attr_name)
        return result

class HQuery(PDH_HQUERY):
    def __init__(self):
        super(HQuery, self).__init__()
        if PdhOpenQueryW(None, None, byref(self)) != PDH_OK:
            raise PDHError("Couldn't open PDH query.")

    def add_counter(self, path: str) -> HCounter:
        hCounter = HCounter()
        if PdhAddEnglishCounterW(self, LPCWSTR(path), None, byref(hCounter)) != PDH_OK:
            raise PDHError("Couldn't add counter query.")
        return hCounter

    def collect_data(self):
        if PdhCollectQueryData(self) != PDH_OK:
            raise PDHError("Couldn't collect query data.")

    def close(self):
        if PdhCloseQuery(self) != PDH_OK:
            raise PDHError("Couldn't close PDH query.")
