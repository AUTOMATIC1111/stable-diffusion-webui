from ctypes import *

msvcrt = CDLL("msvcrt")

malloc = msvcrt.malloc
malloc.restype = c_void_p
malloc.argtypes = [c_size_t]

free = msvcrt.free
free.restype = None
free.argtypes = [c_void_p]
