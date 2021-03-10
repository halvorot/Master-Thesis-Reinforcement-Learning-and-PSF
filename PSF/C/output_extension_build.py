from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("float pi_approx(int n);")

ffibuilder.set_source("_pi",  # name of the output C extension
"""
    #include "output.h"
""",
    sources=['output.c'],   # includes output.c as additional sources
    libraries=[])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)