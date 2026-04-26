#
#   Cython wrapper for the cheesefinder API
#

from typing import Callable

cdef extern from "cheesefinder.h":
    ctypedef void (*cheesefunc)(char *name, void *user_data)
    void find_cheeses(cheesefunc user_func, void *user_data)
    ctypedef int (*cheese_progress_callback)(float progress, void * user_data)
    ctypedef struct cheese_params:
        int age
        cheese_progress_callback progress_callback
        void * user_data

cdef void callback(char *name, void *f) noexcept:
    (<object>f)(name.decode('utf-8'))

def find(f):
    find_cheeses(callback, <void*>f)


cdef int progress_callback(float progress, void * py_progress_callback) noexcept:
    """cheese_progress_callback callback wrapper enabling python callbacks to be used"""
    return (<object>py_progress_callback)(progress)

def py_progress_callback(float progress) -> int:
    print(f"progress: {progress * 100}")
    return 0


cdef class CheeseParams:
    cdef cheese_params p

    def __init__(self, callback: Optional[Callable[[float], int]] = None):
        self.p.age = 10
        self.p.progress_callback = progress_callback
        if callback:
            self.p.user_data = <void*>callback
        else:
            self.p.user_data = <void*>py_progress_callback

    def call(self, float progress, object callback = None):
        if callback:
            self.p.progress_callback(progress, <void*>callback)
        else:
            self.p.progress_callback(progress, <void*>self.p.user_data)

    @property
    def age(self) -> int:
        return self.p.age

    @age.setter
    def age(self, value: int):
        self.p.age = value

    @property
    def callback(self) -> object:
        return <object>self.p.user_data

    @callback.setter
    def callback(self, object cb):
        self.p.user_data = <void*>cb

