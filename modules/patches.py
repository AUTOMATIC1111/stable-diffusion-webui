from collections import defaultdict
from typing import Optional


def patch(key, obj, field, replacement, add_if_not_exists:bool = False):
    """Replaces a function in a module or a class.
    Also stores the original function in this module, possible to be retrieved via original(key, obj, field).
    If the function is already replaced by this caller (key), an exception is raised -- use undo() before that.
    Arguments:
        key: identifying information for who is doing the replacement. You can use __name__.
        obj: the module or the class
        field: name of the function as a string
        replacement: the new function
    Returns:
        the original function
    """
    patch_key = (obj, field)
    if patch_key in originals[key]:
        raise RuntimeError(f"patch for {field} is already applied")
    if not hasattr(obj, field) and not add_if_not_exists:
        raise AttributeError(f"type {type(obj)} '{type.__name__}' has no attribute '{field}'")
    original_func = getattr(obj, field, None)
    originals[key][patch_key] = original_func
    setattr(obj, field, replacement)
    return original_func


def undo(key, obj, field):
    """Undoes the peplacement by the patch().
    If the function is not replaced, raises an exception.
    Arguments:
        key: identifying information for who is doing the replacement. You can use __name__.
        obj: the module or the class
        field: name of the function as a string
    Returns:
        Always None
    """
    patch_key = (obj, field)
    if patch_key not in originals[key]:
        raise RuntimeError(f"there is no patch for {field} to undo")
    original_func = originals[key].pop(patch_key)
    if original_func is None:
        delattr(obj, field)
    setattr(obj, field, original_func)
    return None


def original(key, obj, field):
    """Returns the original function for the patch created by the patch() function"""
    patch_key = (obj, field)
    return originals[key].get(patch_key, None)


def patch_method(cls, key:Optional[str]=None):
    def decorator(func):
        patch(func.__module__ if key is None else key, cls, func.__name__, func)
    return decorator


def add_method(cls, key:Optional[str]=None):
    def decorator(func):
        patch(func.__module__ if key is None else key, cls, func.__name__, func, True)
    return decorator


originals = defaultdict(dict)
