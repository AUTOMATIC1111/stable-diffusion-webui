# Copyright (c) Facebook, Inc. and its affiliates.
""" Utilities for developers only.
These are not visible to users (not automatically imported). And should not
appeared in docs."""
# adapted from https://github.com/tensorpack/tensorpack/blob/master/tensorpack/utils/develop.py


def create_dummy_class(klass, dependency, message=""):
    """
    When a dependency of a class is not available, create a dummy class which throws ImportError
    when used.

    Args:
        klass (str): name of the class.
        dependency (str): name of the dependency.
        message: extra message to print
    Returns:
        class: a class object
    """
    err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, klass)
    if message:
        err = err + " " + message

    class _DummyMetaClass(type):
        # throw error on class attribute access
        def __getattr__(_, __):  # noqa: B902
            raise ImportError(err)

    class _Dummy(object, metaclass=_DummyMetaClass):
        # throw error on constructor
        def __init__(self, *args, **kwargs):
            raise ImportError(err)

    return _Dummy


def create_dummy_func(func, dependency, message=""):
    """
    When a dependency of a function is not available, create a dummy function which throws
    ImportError when used.

    Args:
        func (str): name of the function.
        dependency (str or list[str]): name(s) of the dependency.
        message: extra message to print
    Returns:
        function: a function object
    """
    err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, func)
    if message:
        err = err + " " + message

    if isinstance(dependency, (list, tuple)):
        dependency = ",".join(dependency)

    def _dummy(*args, **kwargs):
        raise ImportError(err)

    return _dummy
