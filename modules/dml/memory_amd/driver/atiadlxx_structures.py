import ctypes as C

class _ADLPMActivity(C.Structure):
    __slot__ = [
        'iActivityPercent',
        'iCurrentBusLanes',
        'iCurrentBusSpeed',
        'iCurrentPerformanceLevel',
        'iEngineClock',
        'iMaximumBusLanes',
        'iMemoryClock',
        'iReserved',
        'iSize',
        'iVddc',
    ]
_ADLPMActivity._fields_ = [
    ('iActivityPercent', C.c_int),
    ('iCurrentBusLanes', C.c_int),
    ('iCurrentBusSpeed', C.c_int),
    ('iCurrentPerformanceLevel', C.c_int),
    ('iEngineClock', C.c_int),
    ('iMaximumBusLanes', C.c_int),
    ('iMemoryClock', C.c_int),
    ('iReserved', C.c_int),
    ('iSize', C.c_int),
    ('iVddc', C.c_int),
]
ADLPMActivity = _ADLPMActivity

class _ADLMemoryInfo2(C.Structure):
    __slot__ = [
        'iHyperMemorySize',
        'iInvisibleMemorySize',
        'iMemoryBandwidth',
        'iMemorySize',
        'iVisibleMemorySize',
        'strMemoryType'
    ]
_ADLMemoryInfo2._fields_ = [
    ('iHyperMemorySize', C.c_longlong),
    ('iInvisibleMemorySize', C.c_longlong),
    ('iMemoryBandwidth', C.c_longlong),
    ('iMemorySize', C.c_longlong),
    ('iVisibleMemorySize', C.c_longlong),
    ('strMemoryType', C.c_char * 256)
]
ADLMemoryInfo2 = _ADLMemoryInfo2

class _AdapterInfo(C.Structure):
    __slot__ = [
        'iSize',
        'iAdapterIndex',
        'strUDID',
        'iBusNumber',
        'iDeviceNumber',
        'iFunctionNumber',
        'iVendorID',
        'strAdapterName',
        'strDisplayName',
        'iPresent',
        'iExist',
        'strDriverPath',
        'strDriverPathExt',
        'strPNPString',
        'iOSDisplayIndex',
    ]
_AdapterInfo._fields_ = [
    ('iSize', C.c_int),
    ('iAdapterIndex', C.c_int),
    ('strUDID', C.c_char * 256),
    ('iBusNumber', C.c_int),
    ('iDeviceNumber', C.c_int),
    ('iFunctionNumber', C.c_int),
    ('iVendorID', C.c_int),
    ('strAdapterName', C.c_char * 256),
    ('strDisplayName', C.c_char * 256),
    ('iPresent', C.c_int),
    ('iExist', C.c_int),
    ('strDriverPath', C.c_char * 256),
    ('strDriverPathExt', C.c_char * 256),
    ('strPNPString', C.c_char * 256),
    ('iOSDisplayIndex', C.c_int)
]
AdapterInfo = _AdapterInfo
LPAdapterInfo = C.POINTER(_AdapterInfo)

ADL_CONTEXT_HANDLE = C.c_void_p