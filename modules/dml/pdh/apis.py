from ctypes import CDLL, POINTER
from ctypes.wintypes import LPCWSTR, LPDWORD, DWORD
from typing import Callable
from .structures import PDH_HQUERY, PDH_HCOUNTER, PPDH_FMT_COUNTERVALUE, PPDH_FMT_COUNTERVALUE_ITEM_W
from .defines import PDH_FUNCTION, PZZWSTR, DWORD_PTR


pdh = CDLL("pdh.dll")


PdhExpandWildCardPathW: Callable = pdh.PdhExpandWildCardPathW
PdhExpandWildCardPathW.restype = PDH_FUNCTION
PdhExpandWildCardPathW.argtypes = [LPCWSTR, LPCWSTR, PZZWSTR, LPDWORD, DWORD]

PdhOpenQueryW: Callable = pdh.PdhOpenQueryW
PdhOpenQueryW.restype = PDH_FUNCTION
PdhOpenQueryW.argtypes = [LPCWSTR, DWORD_PTR, POINTER(PDH_HQUERY)]

PdhAddEnglishCounterW: Callable = pdh.PdhAddEnglishCounterW
PdhAddEnglishCounterW.restype = PDH_FUNCTION
PdhAddEnglishCounterW.argtypes = [PDH_HQUERY, LPCWSTR, DWORD_PTR, POINTER(PDH_HCOUNTER)]

PdhCollectQueryData: Callable = pdh.PdhCollectQueryData
PdhCollectQueryData.restype = PDH_FUNCTION
PdhCollectQueryData.argtypes = [PDH_HQUERY]

PdhGetFormattedCounterValue: Callable = pdh.PdhGetFormattedCounterValue
PdhGetFormattedCounterValue.restype = PDH_FUNCTION
PdhGetFormattedCounterValue.argtypes = [PDH_HCOUNTER, DWORD, LPDWORD, PPDH_FMT_COUNTERVALUE]

PdhGetFormattedCounterArrayW: Callable = pdh.PdhGetFormattedCounterArrayW
PdhGetFormattedCounterArrayW.restype = PDH_FUNCTION
PdhGetFormattedCounterArrayW.argtypes = [PDH_HCOUNTER, DWORD, LPDWORD, LPDWORD, PPDH_FMT_COUNTERVALUE_ITEM_W]

PdhCloseQuery: Callable = pdh.PdhCloseQuery
PdhCloseQuery.restype = PDH_FUNCTION
PdhCloseQuery.argtypes = [PDH_HQUERY]
