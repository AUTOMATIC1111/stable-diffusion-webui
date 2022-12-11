"""Python Advanced Enumerations & NameTuples"""
from __future__ import print_function

# imports
import sys as _sys
pyver = _sys.version_info[:2]
PY2 = pyver < (3, )
PY3 = pyver >= (3, )
PY2_6 = (2, 6)
PY3_3 = (3, 3)
PY3_4 = (3, 4)
PY3_5 = (3, 5)
PY3_6 = (3, 6)
PY3_11 = (3, 11)

import re

_bltin_property = property
_bltin_bin = bin

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
from collections import defaultdict
try:
    import sqlite3
except ImportError:
    sqlite3 = None

try:
    RecursionError
except NameError:
    # python3.4
    RecursionError = RuntimeError

from operator import or_ as _or_, and_ as _and_, xor as _xor_, inv as _inv_
from operator import abs as _abs_, add as _add_, floordiv as _floordiv_
from operator import lshift as _lshift_, rshift as _rshift_, mod as _mod_
from operator import mul as _mul_, neg as _neg_, pos as _pos_, pow as _pow_
from operator import truediv as _truediv_, sub as _sub_

if PY2:
    from ._py2 import *
if PY3:
    from ._py3 import *

obj_type = type

__all__ = [
        'NamedConstant', 'constant', 'skip', 'nonmember', 'member', 'no_arg',
        'Enum', 'IntEnum', 'AutoNumberEnum', 'OrderedEnum', 'UniqueEnum',
        'StrEnum', 'UpperStrEnum', 'LowerStrEnum',
        'Flag', 'IntFlag',
        'AddValue', 'MagicValue', 'MultiValue', 'NoAlias', 'Unique',
        'AddValueEnum', 'MultiValueEnum', 'NoAliasEnum',
        'enum', 'extend_enum', 'unique', 'property',
        'NamedTuple', 'SqliteEnum',
        'FlagBoundary', 'STRICT', 'CONFORM', 'EJECT', 'KEEP',
        'add_stdlib_integration', 'remove_stdlib_integration'
        ]

if sqlite3 is None:
    __all__.remove('SqliteEnum')

version = 3, 1, 12, 1

# shims
try:
    any
except NameError:
    def any(iterable):
        for element in iterable:
            if element:
                return True
        return False

try:
    unicode
    unicode = unicode
except NameError:
    # In Python 3 unicode no longer exists (it's just str)
    unicode = str

try:
    basestring
    basestring = bytes, unicode
except NameError:
    # In Python 2 basestring is the ancestor of both str and unicode
    # in Python 3 it's just str, but was missing in 3.1
    basestring = str,

try:
    long
    baseinteger = int, long
except NameError:
    baseinteger = int,
    long = int
# deprecated
baseint = baseinteger

try:
    NoneType
except NameError:
    NoneType = type(None)

try:
    # derive from stdlib enum if possible
    import enum
    if hasattr(enum, 'version'):
        raise ImportError('wrong version')
    else:
        from enum import EnumMeta as StdlibEnumMeta, Enum as StdlibEnum, IntEnum as StdlibIntEnum
        StdlibFlag = StdlibIntFlag = StdlibStrEnum = StdlibReprEnum = None
except ImportError:
    StdlibEnumMeta = StdlibEnum = StdlibIntEnum = StdlibIntFlag = StdlibFlag = StdlibStrEnum = None

if StdlibEnum:
    try:
        from enum import IntFlag as StdlibIntFlag, Flag as StdlibFlag
    except ImportError:
        pass
    try:
        from enum import StrEnum as StdlibStrEnum
    except ImportError:
        pass
    try:
        from enum import ReprEnum as StdlibReprEnum
    except ImportError:
        pass




# helpers
# will be exported later
MagicValue = AddValue = MultiValue = NoAlias = Unique = None

def _bit_count(num):
    """
    return number of set bits

    Counting bits set, Brian Kernighan's way*

        unsigned int v;          // count the number of bits set in v
        unsigned int c;          // c accumulates the total bits set in v
        for (c = 0; v; c++)
        {   v &= v - 1;  }       //clear the least significant bit set

    This method goes through as many iterations as there are set bits. So if we
    have a 32-bit word with only the high bit set, then it will only go once
    through the loop.

    * The C Programming Language 2nd Ed., Kernighan & Ritchie, 1988.

    This works because each subtraction "borrows" from the lowest 1-bit. For
    example:

          loop pass 1     loop pass 2
          -----------     -----------
               101000          100000
             -      1        -      1
             = 100111        = 011111
             & 101000        & 100000
             = 100000        =      0

    It is an excellent technique for Python, since the size of the integer need
    not be determined beforehand.

    (from https://wiki.python.org/moin/BitManipulation)
    """
    count = 0
    while num:
        num &= num - 1
        count += 1
    return count

def _is_single_bit(value):
    """
    True if only one bit set in value (should be an int)
    """
    if value == 0:
        return False
    value &= value - 1
    return value == 0

def _iter_bits_lsb(value):
    """
    Return each bit value one at a time.

    >>> list(_iter_bits_lsb(6))
    [2, 4]
    """

    while value:
        bit = value & (~value + 1)
        yield bit
        value ^= bit

def bin(value, max_bits=None):
    """
    Like built-in bin(), except negative values are represented in
    twos-compliment, and the leading bit always indicates sign
    (0=positive, 1=negative).

    >>> bin(10)
    '0b0 1010'
    >>> bin(~10)   # ~10 is -11
    '0b1 0101'
    """

    ceiling = 2 ** (value).bit_length()
    if value >= 0:
        s = _bltin_bin(value + ceiling).replace('1', '0', 1)
    else:
        s = _bltin_bin(~value ^ (ceiling - 1) + ceiling)
    sign = s[:3]
    digits = s[3:]
    if max_bits is not None:
        if len(digits) < max_bits:
            digits = (sign[-1] * max_bits + digits)[-max_bits:]
    return "%s %s" % (sign, digits)


try:
    from types import DynamicClassAttribute
    base = DynamicClassAttribute
except ImportError:
    base = object
    DynamicClassAttribute = None

class property(base):
    """
    This is a descriptor, used to define attributes that act differently
    when accessed through an enum member and through an enum class.
    Instance access is the same as property(), but access to an attribute
    through the enum class will look in the class' _member_map_.
    """

    # inherit from DynamicClassAttribute if we can in order to get `inspect`
    # support

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        # next two lines make property act the same as _bltin_property
        self.__doc__ = doc or fget.__doc__
        self.overwrite_doc = doc is None
        # support for abstract methods
        self.__isabstractmethod__ = bool(getattr(fget, '__isabstractmethod__', False))
        # names, if possible

    def getter(self, fget):
        fdoc = fget.__doc__ if self.overwrite_doc else None
        result = type(self)(fget, self.fset, self.fdel, fdoc or self.__doc__)
        result.overwrite_doc = self.__doc__ is None
        return result

    def setter(self, fset):
        fdoc = fget.__doc__ if self.overwrite_doc else None
        result = type(self)(self.fget, fset, self.fdel, self.__doc__)
        result.overwrite_doc = self.__doc__ is None
        return result

    def deleter(self, fdel):
        fdoc = fget.__doc__ if self.overwrite_doc else None
        result = type(self)(self.fget, self.fset, fdel, self.__doc__)
        result.overwrite_doc = self.__doc__ is None
        return result

    def __repr__(self):
        member = self.ownerclass._member_map_.get(self.name)
        func = self.fget or self.fset or self.fdel
        strings = []
        if member:
            strings.append('%r' % member)
        if func:
            strings.append('function=%s' % func.__name__)
        return 'property(%s)' % ', '.join(strings)

    def __get__(self, instance, ownerclass=None):
        if instance is None:
            try:
                return ownerclass._member_map_[self.name]
            except KeyError:
                raise AttributeError(
                        '%r has no attribute %r' % (ownerclass, self.name)
                        )
        else:
            if self.fget is not None:
                return self.fget(instance)
            else:
                if self.fset is not None:
                    raise AttributeError(
                            'cannot read attribute %r on %r' % (self.name, ownerclass)
                            )
                else:
                    try:
                        return instance.__dict__[self.name]
                    except KeyError:
                        raise AttributeError(
                                '%r member has no attribute %r' % (ownerclass, self.name)
                                )

    def __set__(self, instance, value):
        if self.fset is None:
            if self.fget is not None:
                raise AttributeError(
                        "cannot set attribute %r on <aenum %r>" % (self.name, self.clsname)
                        )
            else:
                instance.__dict__[self.name] = value
        else:
            return self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            if self.fget or self.fset:
                raise AttributeError(
                        "cannot delete attribute %r on <aenum %r>" % (self.name, self.clsname)
                        )
            elif self.name in instance.__dict__:
                del instance.__dict__[self.name]
            else:
                raise AttributeError(
                        "no attribute %r on <aenum %r> member" % (self.name, self.clsname)
                        )
        else:
            return self.fdel(instance)

    def __set_name__(self, ownerclass, name):
        self.name = name
        self.clsname = ownerclass.__name__
        self.ownerclass = ownerclass

_RouteClassAttributeToGetattr = property
if DynamicClassAttribute is None:
    DynamicClassAttribute = property
# deprecated
enum_property = property

class NonMember(object):
    """
    Protects item from becaming an Enum member during class creation.
    """
    def __init__(self, value):
        self.value = value

    def __get__(self, instance, ownerclass=None):
        return self.value
skip = nonmember = NonMember

class Member(object):
    """
    Forces item to became an Enum member during class creation.
    """
    def __init__(self, value):
        self.value = value
member = Member

class SentinelType(type):
    def __repr__(cls):
        return '<%s>' % cls.__name__
Sentinel = SentinelType('Sentinel', (object, ), {})

def _is_descriptor(obj):
    """Returns True if obj is a descriptor, False otherwise."""
    return (
            hasattr(obj, '__get__') or
            hasattr(obj, '__set__') or
            hasattr(obj, '__delete__'))


def _is_dunder(name):
    """Returns True if a __dunder__ name, False otherwise."""
    return (len(name) > 4 and
            name[:2] == name[-2:] == '__' and
            name[2] != '_' and
            name[-3] != '_')


def _is_sunder(name):
    """Returns True if a _sunder_ name, False otherwise."""
    return (len(name) > 2 and
            name[0] == name[-1] == '_' and
            name[1] != '_' and
            name[-2] != '_')

def _is_internal_class(cls_name, obj):
    # only 3.3 and up, always return False in 3.2 and below
    if pyver < PY3_3:
        return False
    else:
        qualname = getattr(obj, '__qualname__', False)
        return not _is_descriptor(obj) and qualname and re.search(r"\.?%s\.\w+$" % cls_name, qualname)

def _is_private_name(cls_name, name):
    pattern = r'^_%s__\w+[^_]_?$' % (cls_name, )
    return re.search(pattern, name)

def _power_of_two(value):
    if value < 1:
        return False
    return value == 2 ** _high_bit(value)

def bits(num):
    if num in (0, 1):
        return str(num)
    negative = False
    if num < 0:
        negative = True
        num = ~num
    result = bits(num>>1) + str(num&1)
    if negative:
        result = '1' + ''.join(['10'[d=='1'] for d in result])
    return result


def bit_count(num):
    """
        return number of set bits

        Counting bits set, Brian Kernighan's way*

            unsigned int v;          // count the number of bits set in v
            unsigned int c;          // c accumulates the total bits set in v
            for (c = 0; v; c++)
            {   v &= v - 1;  }       //clear the least significant bit set

        This method goes through as many iterations as there are set bits. So if we
        have a 32-bit word with only the high bit set, then it will only go once
        through the loop.

        * The C Programming Language 2nd Ed., Kernighan & Ritchie, 1988.

        This works because each subtraction "borrows" from the lowest 1-bit. For example:

              loop pass 1     loop pass 2
              -----------     -----------
                   101000          100000
                 -      1        -      1
                 = 100111        = 011111
                 & 101000        & 100000
                 = 100000        =      0

        It is an excellent technique for Python, since the size of the integer need not
        be determined beforehand.
    """
    count = 0
    while(num):
        num &= num - 1
        count += 1
    return(count)

def bit_len(num):
    length = 0
    while num:
        length += 1
        num >>= 1
    return length

def is_single_bit(num):
    """
    True if only one bit set in num (should be an int)
    """
    num &= num - 1
    return num == 0

def _make_class_unpicklable(obj):
    """
    Make the given obj un-picklable.

    obj should be either a dictionary, on an Enum
    """
    def _break_on_call_reduce(self, proto):
        raise TypeError('%r cannot be pickled' % self)
    if isinstance(obj, dict):
        obj['__reduce_ex__'] = _break_on_call_reduce
        obj['__module__'] = '<unknown>'
    else:
        setattr(obj, '__reduce_ex__', _break_on_call_reduce)
        setattr(obj, '__module__', '<unknown>')

def _check_auto_args(method):
    """check if new generate method supports *args and **kwds"""
    if isinstance(method, staticmethod):
        method = method.__get__(type)
    method = getattr(method, 'im_func', method)
    args, varargs, keywords, defaults = getargspec(method)
    return varargs is not None and keywords is not None

def _get_attr_from_chain(cls, attr):
    sentinel = object()
    for basecls in cls.mro():
        obj = basecls.__dict__.get(attr, sentinel)
        if obj is not sentinel:
            return obj

def _value(obj):
    if isinstance(obj, (auto, constant)):
        return obj.value
    else:
        return obj

def enumsort(things):
    """
    sorts things by value if all same type; otherwise by name
    """
    if not things:
        return things
    sort_type = type(things[0])
    if not issubclass(sort_type, tuple):
        # direct sort or type error
        if not all((type(v) is sort_type) for v in things[1:]):
            raise TypeError('cannot sort items of different types')
        return sorted(things)
    else:
        # expecting list of (name, value) tuples
        sort_type = type(things[0][1])
        try:
            if all((type(v[1]) is sort_type) for v in things[1:]):
                return sorted(things, key=lambda i: i[1])
            else:
                raise TypeError('try name sort instead')
        except TypeError:
            return sorted(things, key=lambda i: i[0])

def export(collection, namespace=None):
    """
    export([collection,] namespace) -> Export members to target namespace.

    If collection is not given, act as a decorator.
    """
    if namespace is None:
        namespace = collection
        def export_decorator(collection):
            return export(collection, namespace)
        return export_decorator
    elif issubclass(collection, NamedConstant):
        for n, c in collection.__dict__.items():
            if isinstance(c, NamedConstant):
                namespace[n] = c
    elif issubclass(collection, Enum):
        data = collection.__members__.items()
        for n, m in data:
            namespace[n] = m
    else:
        raise TypeError('%r is not a supported collection' % (collection,) )
    return collection

class _Addendum(object):
    def __init__(self, dict, doc, ns):
        # dict is the dict to update with functions
        # doc is the docstring to put in the dict
        # ns is the namespace to remove the function names from
        self.dict = dict
        self.ns = ns
        self.added = set()
    def __call__(self, func):
        if isinstance(func, (staticmethod, classmethod)):
            name = func.__func__.__name__
        elif isinstance(func, (property, _bltin_property)):
            name = (func.fget or func.fset or func.fdel).__name__
        else:
            name = func.__name__
        self.dict[name] = func
        self.added.add(name)
    def resolve(self):
        ns = self.ns
        for name in self.added:
            del ns[name]
        return self.dict

# Constant / NamedConstant

class constant(object):
    '''
    Simple constant descriptor for NamedConstant and Enum use.
    '''
    def __init__(self, value, doc=None):
        self.value = value
        self.__doc__ = doc

    def __get__(self, *args):
        return self.value

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.value)

    def __and__(self, other):
        return _and_(self.value, _value(other))

    def __rand__(self, other):
        return _and_(_value(other), self.value)

    def __invert__(self):
        return _inv_(self.value)

    def __or__(self, other):
        return _or_(self.value, _value(other))

    def __ror__(self, other):
        return _or_(_value(other), self.value)

    def __xor__(self, other):
        return _xor_(self.value, _value(other))

    def __rxor__(self, other):
        return _xor_(_value(other), self.value)

    def __abs__(self):
        return _abs_(self.value)

    def __add__(self, other):
        return _add_(self.value, _value(other))

    def __radd__(self, other):
        return _add_(_value(other), self.value)

    def __neg__(self):
        return _neg_(self.value)

    def __pos__(self):
        return _pos_(self.value)

    if PY2:
        def __div__(self, other):
            return _div_(self.value, _value(other))

    def __rdiv__(self, other):
        return _div_(_value(other), (self.value))

    def __floordiv__(self, other):
        return _floordiv_(self.value, _value(other))

    def __rfloordiv__(self, other):
        return _floordiv_(_value(other), self.value)

    def __truediv__(self, other):
        return _truediv_(self.value, _value(other))

    def __rtruediv__(self, other):
        return _truediv_(_value(other), self.value)

    def __lshift__(self, other):
        return _lshift_(self.value, _value(other))

    def __rlshift__(self, other):
        return _lshift_(_value(other), self.value)

    def __rshift__(self, other):
        return _rshift_(self.value, _value(other))

    def __rrshift__(self, other):
        return _rshift_(_value(other), self.value)

    def __mod__(self, other):
        return _mod_(self.value, _value(other))

    def __rmod__(self, other):
        return _mod_(_value(other), self.value)

    def __mul__(self, other):
        return _mul_(self.value, _value(other))

    def __rmul__(self, other):
        return _mul_(_value(other), self.value)

    def __pow__(self, other):
        return _pow_(self.value, _value(other))

    def __rpow__(self, other):
        return _pow_(_value(other), self.value)

    def __sub__(self, other):
        return _sub_(self.value, _value(other))

    def __rsub__(self, other):
        return _sub_(_value(other), self.value)

    def __set_name__(self, ownerclass, name):
        self.name = name
        self.clsname = ownerclass.__name__


NamedConstant = None

class _NamedConstantDict(dict):
    """Track constant order and ensure names are not reused.

    NamedConstantMeta will use the names found in self._names as the
    Constant names.
    """
    def __init__(self):
        super(_NamedConstantDict, self).__init__()
        self._names = []

    def __setitem__(self, key, value):
        """Changes anything not dundered or not a constant descriptor.

        If an constant name is used twice, an error is raised; duplicate
        values are not checked for.

        Single underscore (sunder) names are reserved.
        """
        if _is_sunder(key):
            raise ValueError(
                    '_sunder_ names, such as %r, are reserved for future NamedConstant use'
                    % (key, )
                    )
        elif _is_dunder(key):
            pass
        elif key in self._names:
            # overwriting an existing constant?
            raise TypeError('attempt to reuse name: %r' % (key, ))
        elif isinstance(value, constant) or not _is_descriptor(value):
            if key in self:
                # overwriting a descriptor?
                raise TypeError('%s already defined as: %r' % (key, self[key]))
            self._names.append(key)
        super(_NamedConstantDict, self).__setitem__(key, value)


class NamedConstantMeta(type):
    """
    Block attempts to reassign NamedConstant attributes.
    """

    @classmethod
    def __prepare__(metacls, cls, bases, **kwds):
        return _NamedConstantDict()

    def __new__(metacls, cls, bases, clsdict):
        if type(clsdict) is dict:
            original_dict = clsdict
            clsdict = _NamedConstantDict()
            for k, v in original_dict.items():
                clsdict[k] = v
        newdict = {}
        constants = {}
        for name, obj in clsdict.items():
            if name in clsdict._names:
                constants[name] = obj
                continue
            elif isinstance(obj, nonmember):
                obj = obj.value
            newdict[name] = obj
        newcls = super(NamedConstantMeta, metacls).__new__(metacls, cls, bases, newdict)
        newcls._named_constant_cache_ = {}
        newcls._members_ = {}
        for name, obj in constants.items():
            new_k = newcls.__new__(newcls, name, obj)
            newcls._members_[name] = new_k
        return newcls

    def __bool__(cls):
        return True

    def __delattr__(cls, attr):
        cur_obj = cls.__dict__.get(attr)
        if NamedConstant is not None and isinstance(cur_obj, NamedConstant):
            raise AttributeError('cannot delete constant <%s.%s>' % (cur_obj.__class__.__name__, cur_obj._name_))
        super(NamedConstantMeta, cls).__delattr__(attr)

    def __iter__(cls):
        return (k for k in cls._members_.values())

    def __reversed__(cls):
        return (k for k in reversed(cls._members_.values()))

    def __len__(cls):
        return len(cls._members_)

    __nonzero__ = __bool__

    def __setattr__(cls, name, value):
        """Block attempts to reassign NamedConstants.
        """
        cur_obj = cls.__dict__.get(name)
        if NamedConstant is not None and isinstance(cur_obj, NamedConstant):
            raise AttributeError('cannot rebind constant <%s.%s>' % (cur_obj.__class__.__name__, cur_obj._name_))
        super(NamedConstantMeta, cls).__setattr__(name, value)

constant_dict = _Addendum(
        dict=NamedConstantMeta.__prepare__('NamedConstant', (object, )),
        doc="NamedConstants protection.\n\n    Derive from this class to lock NamedConstants.\n\n",
        ns=globals(),
        )

@constant_dict
def __new__(cls, name, value=None, doc=None):
    if value is None:
        # lookup, name is value
        value = name
        for name, obj in cls.__dict__.items():
            if isinstance(obj, cls) and obj._value_ == value:
                return obj
        else:
            raise ValueError('%r does not exist in %r' % (value, cls.__name__))
    cur_obj = cls.__dict__.get(name)
    if isinstance(cur_obj, NamedConstant):
        raise AttributeError('cannot rebind constant <%s.%s>' % (cur_obj.__class__.__name__, cur_obj._name_))
    elif isinstance(value, constant):
        doc = doc or value.__doc__
        value = value.value
    metacls = cls.__class__
    if isinstance(value, NamedConstant):
        # constants from other classes are reduced to their actual value
        value = value._value_
    actual_type = type(value)
    value_type = cls._named_constant_cache_.get(actual_type)
    if value_type is None:
        value_type = type(cls.__name__, (cls, type(value)), {})
        cls._named_constant_cache_[type(value)] = value_type
    obj = actual_type.__new__(value_type, value)
    obj._name_ = name
    obj._value_ = value
    obj.__doc__ = doc
    cls._members_[name] = obj
    metacls.__setattr__(cls, name, obj)
    return obj

@constant_dict
def __repr__(self):
    return "<%s.%s: %r>" % (
            self.__class__.__name__, self._name_, self._value_)

@constant_dict
def __reduce_ex__(self, proto):
    return getattr, (self.__class__, self._name_)

NamedConstant = NamedConstantMeta('NamedConstant', (object, ), constant_dict.resolve())
Constant = NamedConstant
del constant_dict

# NamedTuple

class _NamedTupleDict(OrderedDict):
    """Track field order and ensure field names are not reused.

    NamedTupleMeta will use the names found in self._field_names to translate
    to indices.
    """
    def __init__(self, *args, **kwds):
        self._field_names = []
        super(_NamedTupleDict, self).__init__(*args, **kwds)

    def __setitem__(self, key, value):
        """Records anything not dundered or not a descriptor.

        If a field name is used twice, an error is raised.

        Single underscore (sunder) names are reserved.
        """
        if _is_sunder(key):
            if key not in ('_size_', '_order_', '_fields_'):
                raise ValueError(
                        '_sunder_ names, such as %r, are reserved for future NamedTuple use'
                        % (key, )
                        )
        elif _is_dunder(key):
            if key == '__order__':
                key = '_order_'
        elif key in self._field_names:
            # overwriting a field?
            raise TypeError('attempt to reuse field name: %r' % (key, ))
        elif not _is_descriptor(value):
            if key in self:
                # field overwriting a descriptor?
                raise TypeError('%s already defined as: %r' % (key, self[key]))
            self._field_names.append(key)
        super(_NamedTupleDict, self).__setitem__(key, value)


class _TupleAttributeAtIndex(object):

    def __init__(self, name, index, doc, default):
        self.name = name
        self.index = index
        if doc is undefined:
            doc = None
        self.__doc__ = doc
        self.default = default

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if len(instance) <= self.index:
            raise AttributeError('%s instance has no value for %s' % (instance.__class__.__name__, self.name))
        return instance[self.index]

    def __repr__(self):
        return '%s(%d)' % (self.__class__.__name__, self.index)


class undefined(object):
    def __repr__(self):
        return 'undefined'
    def __bool__(self):
        return False
    __nonzero__ = __bool__
undefined = undefined()


class TupleSize(NamedConstant):
    fixed = constant('fixed', 'tuple length is static')
    minimum = constant('minimum', 'tuple must be at least x long (x is calculated during creation')
    variable = constant('variable', 'tuple length can be anything')

class NamedTupleMeta(type):
    """Metaclass for NamedTuple"""

    @classmethod
    def __prepare__(metacls, cls, bases, size=undefined, **kwds):
        return _NamedTupleDict()

    def __init__(cls, *args , **kwds):
        super(NamedTupleMeta, cls).__init__(*args)

    def __new__(metacls, cls, bases, clsdict, size=undefined, **kwds):
        if bases == (object, ):
            bases = (tuple, object)
        elif tuple not in bases:
            if object in bases:
                index = bases.index(object)
                bases = bases[:index] + (tuple, ) + bases[index:]
            else:
                bases = bases + (tuple, )
        # include any fields from base classes
        base_dict = _NamedTupleDict()
        namedtuple_bases = []
        for base in bases:
            if isinstance(base, NamedTupleMeta):
                namedtuple_bases.append(base)
        i = 0
        if namedtuple_bases:
            for name, index, doc, default in metacls._convert_fields(*namedtuple_bases):
                base_dict[name] = index, doc, default
                i = max(i, index)
        # construct properly ordered dict with normalized indexes
        for k, v in clsdict.items():
            base_dict[k] = v
        original_dict = base_dict
        if size is not undefined and '_size_' in original_dict:
            raise TypeError('_size_ cannot be set if "size" is passed in header')
        add_order = isinstance(clsdict, _NamedTupleDict)
        clsdict = _NamedTupleDict()
        clsdict.setdefault('_size_', size or TupleSize.fixed)
        unnumbered = OrderedDict()
        numbered = OrderedDict()
        _order_ = original_dict.pop('_order_', [])
        if _order_ :
            _order_ = _order_.replace(',',' ').split()
            add_order = False
        # and process this class
        for k, v in original_dict.items():
            if k not in original_dict._field_names:
                clsdict[k] = v
            else:
                # TODO:normalize v here
                if isinstance(v, baseinteger):
                    # assume an offset
                    v = v, undefined, undefined
                    i = v[0] + 1
                    target = numbered
                elif isinstance(v, basestring):
                    # assume a docstring
                    if add_order:
                        v = i, v, undefined
                        i += 1
                        target = numbered
                    else:
                        v = undefined, v, undefined
                        target = unnumbered
                elif isinstance(v, tuple) and len(v) in (2, 3) and isinstance(v[0], baseinteger) and isinstance(v[1], (basestring, NoneType)):
                    # assume an offset, a docstring, and (maybe) a default
                    if len(v) == 2:
                        v = v + (undefined, )
                    v = v
                    i = v[0] + 1
                    target = numbered
                elif isinstance(v, tuple) and len(v) in (1, 2) and isinstance(v[0], (basestring, NoneType)):
                    # assume a docstring, and (maybe) a default
                    if len(v) == 1:
                        v = v + (undefined, )
                    if add_order:
                        v = (i, ) + v
                        i += 1
                        target = numbered
                    else:
                        v = (undefined, ) + v
                        target = unnumbered
                else:
                    # refuse to guess further
                    raise ValueError('not sure what to do with %s=%r (should be OFFSET [, DOC [, DEFAULT]])' % (k, v))
                target[k] = v
        # all index values have been normalized
        # deal with _order_ (or lack thereof)
        fields = []
        aliases = []
        seen = set()
        max_len = 0
        if not _order_:
            if unnumbered:
                raise ValueError("_order_ not specified and OFFSETs not declared for %r" % (unnumbered.keys(), ))
            for name, (index, doc, default) in sorted(numbered.items(), key=lambda nv: (nv[1][0], nv[0])):
                if index in seen:
                    aliases.append(name)
                else:
                    fields.append(name)
                    seen.add(index)
                    max_len = max(max_len, index + 1)
            offsets = numbered
        else:
            # check if any unnumbered not in _order_
            missing = set(unnumbered) - set(_order_)
            if missing:
                raise ValueError("unable to order fields: %s (use _order_ or specify OFFSET" % missing)
            offsets = OrderedDict()
            # if any unnumbered, number them from their position in _order_
            i = 0
            for k in _order_:
                try:
                    index, doc, default = unnumbered.pop(k, None) or numbered.pop(k)
                except IndexError:
                    raise ValueError('%s (from _order_) not found in %s' % (k, cls))
                if index is not undefined:
                    i = index
                if i in seen:
                    aliases.append(k)
                else:
                    fields.append(k)
                    seen.add(i)
                offsets[k] = i, doc, default
                i += 1
                max_len = max(max_len, i)
            # now handle anything in numbered
            for k, (index, doc, default) in sorted(numbered.items(), key=lambda nv: (nv[1][0], nv[0])):
                if index in seen:
                    aliases.append(k)
                else:
                    fields.append(k)
                    seen.add(index)
                offsets[k] = index, doc, default
                max_len = max(max_len, index+1)

        # at this point fields and aliases should be ordered lists, offsets should be an
        # OrdededDict with each value an int, str or None or undefined, default or None or undefined
        assert len(fields) + len(aliases) == len(offsets), "number of fields + aliases != number of offsets"
        assert set(fields) & set(offsets) == set(fields), "some fields are not in offsets: %s" % set(fields) & set(offsets)
        assert set(aliases) & set(offsets) == set(aliases), "some aliases are not in offsets: %s" % set(aliases) & set(offsets)
        for name, (index, doc, default) in offsets.items():
            assert isinstance(index, baseinteger), "index for %s is not an int (%s:%r)" % (name, type(index), index)
            assert isinstance(doc, (basestring, NoneType)) or doc is undefined, "doc is not a str, None, nor undefined (%s:%r)" % (name, type(doc), doc)

        # create descriptors for fields
        for name, (index, doc, default) in offsets.items():
            clsdict[name] = _TupleAttributeAtIndex(name, index, doc, default)
        clsdict['__slots__'] = ()

        # create our new NamedTuple type
        namedtuple_class = super(NamedTupleMeta, metacls).__new__(metacls, cls, bases, clsdict)
        namedtuple_class._fields_ = fields
        namedtuple_class._aliases_ = aliases
        namedtuple_class._defined_len_ = max_len
        return namedtuple_class

    @staticmethod
    def _convert_fields(*namedtuples):
        "create list of index, doc, default triplets for cls in namedtuples"
        all_fields = []
        for cls in namedtuples:
            base = len(all_fields)
            for field in cls._fields_:
                desc = getattr(cls, field)
                all_fields.append((field, base+desc.index, desc.__doc__, desc.default))
        return all_fields

    def __add__(cls, other):
        "A new NamedTuple is created by concatenating the _fields_ and adjusting the descriptors"
        if not isinstance(other, NamedTupleMeta):
            return NotImplemented
        return NamedTupleMeta('%s%s' % (cls.__name__, other.__name__), (cls, other), {})

    def __call__(cls, *args, **kwds):
        """Creates a new NamedTuple class or an instance of a NamedTuple subclass.

        NamedTuple should have args of (class_name, names, module)

            `names` can be:

                * A string containing member names, separated either with spaces or
                  commas.  Values are auto-numbered from 1.
                * An iterable of member names.  Values are auto-numbered from 1.
                * An iterable of (member name, value) pairs.
                * A mapping of member name -> value.

                `module`, if set, will be stored in the new class' __module__ attribute;

                Note: if `module` is not set this routine will attempt to discover the
                calling module by walking the frame stack; if this is unsuccessful
                the resulting class will not be pickleable.

        subclass should have whatever arguments and/or keywords will be used to create an
        instance of the subclass
        """
        if cls is NamedTuple:
            original_args = args
            original_kwds = kwds.copy()
            # create a new subclass
            try:
                if 'class_name' in kwds:
                    class_name = kwds.pop('class_name')
                else:
                    class_name, args = args[0], args[1:]
                if 'names' in kwds:
                    names = kwds.pop('names')
                else:
                    names, args = args[0], args[1:]
                if 'module' in kwds:
                    module = kwds.pop('module')
                elif args:
                    module, args = args[0], args[1:]
                else:
                    module = None
                if 'type' in kwds:
                    type = kwds.pop('type')
                elif args:
                    type, args = args[0], args[1:]
                else:
                    type = None

            except IndexError:
                raise TypeError('too few arguments to NamedTuple: %s, %s' % (original_args, original_kwds))
            if args or kwds:
                raise TypeError('too many arguments to NamedTuple: %s, %s' % (original_args, original_kwds))
            if PY2:
                # if class_name is unicode, attempt a conversion to ASCII
                if isinstance(class_name, unicode):
                    try:
                        class_name = class_name.encode('ascii')
                    except UnicodeEncodeError:
                        raise TypeError('%r is not representable in ASCII' % (class_name, ))
            # quick exit if names is a NamedTuple
            if isinstance(names, NamedTupleMeta):
                names.__name__ = class_name
                if type is not None and type not in names.__bases__:
                    names.__bases__ = (type, ) + names.__bases__
                return names

            metacls = cls.__class__
            bases = (cls, )
            clsdict = metacls.__prepare__(class_name, bases)

            # special processing needed for names?
            if isinstance(names, basestring):
                names = names.replace(',', ' ').split()
            if isinstance(names, (tuple, list)) and isinstance(names[0], basestring):
                names = [(e, i) for (i, e) in enumerate(names)]
            # Here, names is either an iterable of (name, index) or (name, index, doc, default) or a mapping.
            item = None  # in case names is empty
            for item in names:
                if isinstance(item, basestring):
                    # mapping
                    field_name, field_index = item, names[item]
                else:
                    # non-mapping
                    if len(item) == 2:
                        field_name, field_index = item
                    else:
                        field_name, field_index = item[0], item[1:]
                clsdict[field_name] = field_index
            if type is not None:
                if not isinstance(type, tuple):
                    type = (type, )
                bases = type + bases
            namedtuple_class = metacls.__new__(metacls, class_name, bases, clsdict)

            # TODO: replace the frame hack if a blessed way to know the calling
            # module is ever developed
            if module is None:
                try:
                    module = _sys._getframe(1).f_globals['__name__']
                except (AttributeError, ValueError, KeyError):
                    pass
            if module is None:
                _make_class_unpicklable(namedtuple_class)
            else:
                namedtuple_class.__module__ = module

            return namedtuple_class
        else:
            # instantiate a subclass
            namedtuple_instance = cls.__new__(cls, *args, **kwds)
            if isinstance(namedtuple_instance, cls):
                namedtuple_instance.__init__(*args, **kwds)
            return namedtuple_instance

    @_bltin_property
    def __fields__(cls):
        return list(cls._fields_)
    # collections.namedtuple compatibility
    _fields = __fields__

    @_bltin_property
    def __aliases__(cls):
        return list(cls._aliases_)

    def __repr__(cls):
        return "<NamedTuple %r>" % (cls.__name__, )

namedtuple_dict = _Addendum(
        dict=NamedTupleMeta.__prepare__('NamedTuple', (object, )),
        doc="NamedTuple base class.\n\n    Derive from this class to define new NamedTuples.\n\n",
        ns=globals(),
        )

@namedtuple_dict
def __new__(cls, *args, **kwds):
    if cls._size_ is TupleSize.fixed and len(args) > cls._defined_len_:
        raise TypeError('%d fields expected, %d received' % (cls._defined_len_, len(args)))
    unknown = set(kwds) - set(cls._fields_) - set(cls._aliases_)
    if unknown:
        raise TypeError('unknown fields: %r' % (unknown, ))
    final_args = list(args) + [undefined] * (len(cls.__fields__) - len(args))
    for field, value in kwds.items():
        index = getattr(cls, field).index
        if final_args[index] != undefined:
            raise TypeError('field %s specified more than once' % field)
        final_args[index] = value
    missing = []
    for index, value in enumerate(final_args):
        if value is undefined:
            # look for default values
            name = cls.__fields__[index]
            default = getattr(cls, name).default
            if default is undefined:
                missing.append(name)
            else:
                final_args[index] = default
    if missing:
        if cls._size_ in (TupleSize.fixed, TupleSize.minimum):
            raise TypeError('values not provided for field(s): %s' % ', '.join(missing))
        while final_args and final_args[-1] is undefined:
            final_args.pop()
            missing.pop()
        if cls._size_ is not TupleSize.variable or undefined in final_args:
            raise TypeError('values not provided for field(s): %s' % ', '.join(missing))
    return tuple.__new__(cls, tuple(final_args))

@namedtuple_dict
def __reduce_ex__(self, proto):
    return self.__class__, tuple(getattr(self, f) for f in self._fields_)

@namedtuple_dict
def __repr__(self):
    if len(self) == len(self._fields_):
        return "%s(%s)" % (
                self.__class__.__name__, ', '.join(['%s=%r' % (f, o) for f, o in zip(self._fields_, self)])
                )
    else:
        return '%s(%s)' % (self.__class__.__name__, ', '.join([repr(o) for o in self]))

@namedtuple_dict
def __str__(self):
    return "%s(%s)" % (
            self.__class__.__name__, ', '.join(['%r' % (getattr(self, f), ) for f in self._fields_])
            )

@namedtuple_dict
@_bltin_property
def _fields_(self):
    return list(self.__class__._fields_)

    # compatibility methods with stdlib namedtuple
@namedtuple_dict
@_bltin_property
def __aliases__(self):
    return list(self.__class__._aliases_)

@namedtuple_dict
@_bltin_property
def _fields(self):
    return list(self.__class__._fields_)

@namedtuple_dict
@classmethod
def _make(cls, iterable, new=None, len=None):
    return cls.__new__(cls, *iterable)

@namedtuple_dict
def _asdict(self):
    return OrderedDict(zip(self._fields_, self))

@namedtuple_dict
def _replace(self, **kwds):
    current = self._asdict()
    current.update(kwds)
    return self.__class__(**current)

NamedTuple = NamedTupleMeta('NamedTuple', (object, ), namedtuple_dict.resolve())
del namedtuple_dict


# Enum

    # _init_ and value and AddValue
    # -----------------------------
    # by default, when defining a member everything after the = is "the value", everything is
    #   passed to __new__, everything is passed to __init__
    #
    # if _init_ is present then
    #   if `value` is not in _init_, everything is "the value", defaults apply
    #   if `value` is in _init_, only the first thing after the = is the value, and the rest will
    #       be passed to __init__
    #   if fewer values are present for member assignment than _init_ calls for, _generate_next_value_
    #       will be called in an attempt to generate them
    #
    # if AddValue is present then
    #   _generate_next_value_ is always called, and any generated values are prepended to provided
    #       values (custom _gnv_s can change that)
    #   default _init_ rules apply


    # Constants used in Enum

@export(globals())
class EnumConstants(NamedConstant):
    AddValue = constant('addvalue', 'prepends value(s) from _generate_next_value_ to each member')
    MagicValue = constant('magicvalue', 'calls _generate_next_value_ when no arguments are given')
    MultiValue = constant('multivalue', 'each member can have several values')
    NoAlias = constant('noalias', 'duplicate valued members are distinct, not aliased')
    Unique = constant('unique', 'duplicate valued members are not allowed')
    def __repr__(self):
        return self._name_


    # Dummy value for Enum as EnumType explicity checks for it, but of course until
    # EnumType finishes running the first time the Enum class doesn't exist.  This
    # is also why there are checks in EnumType like `if Enum is not None`.
    #
    # Ditto for Flag.

Enum = ReprEnum = IntEnum = StrEnum = Flag = IntFlag = EJECT = KEEP = None

class enum(object):
    """
    Helper class to track args, kwds.
    """
    def __init__(self, *args, **kwds):
        self._args = args
        self._kwds = dict(kwds.items())
        self._hash = hash(args)
        self.name = None

    @_bltin_property
    def args(self):
        return self._args

    @_bltin_property
    def kwds(self):
        return self._kwds.copy()

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.args == other.args and self.kwds == other.kwds

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.args != other.args or self.kwds != other.kwds

    def __repr__(self):
        final = []
        args = ', '.join(['%r' % (a, ) for a in self.args])
        if args:
            final.append(args)
        kwds = ', '.join([('%s=%r') % (k, v) for k, v in enumsort(list(self.kwds.items()))])
        if kwds:
            final.append(kwds)
        return '%s(%s)' % (self.__class__.__name__, ', '.join(final))

_auto_null = SentinelType('no_value', (object, ), {})
class auto(enum):
    """
    Instances are replaced with an appropriate value in Enum class suites.
    """
    enum_member = _auto_null
    _value = _auto_null
    _operations = []

    def __and__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_and_, (self, other)))
        return new_auto

    def __rand__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_and_, (other, self)))
        return new_auto

    def __invert__(self):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_inv_, (self,)))
        return new_auto

    def __or__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_or_, (self, other)))
        return new_auto

    def __ror__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_or_, (other, self)))
        return new_auto

    def __xor__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_xor_, (self, other)))
        return new_auto

    def __rxor__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_xor_, (other, self)))
        return new_auto

    def __abs__(self):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_abs_, (self, )))
        return new_auto

    def __add__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_add_, (self, other)))
        return new_auto

    def __radd__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_add_, (other, self)))
        return new_auto

    def __neg__(self):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_neg_, (self, )))
        return new_auto

    def __pos__(self):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_pos_, (self, )))
        return new_auto

    if PY2:
        def __div__(self, other):
            new_auto = self.__class__()
            new_auto._operations = self._operations[:]
            new_auto._operations.append((_div_, (self, other)))
            return new_auto

    def __rdiv__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_div_, (other, self)))
        return new_auto

    def __floordiv__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_floordiv_, (self, other)))
        return new_auto

    def __rfloordiv__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_floordiv_, (other, self)))
        return new_auto

    def __truediv__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_truediv_, (self, other)))
        return new_auto

    def __rtruediv__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_truediv_, (other, self)))
        return new_auto

    def __lshift__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_lshift_, (self, other)))
        return new_auto

    def __rlshift__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_lshift_, (other, self)))
        return new_auto

    def __rshift__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_rshift_, (self, other)))
        return new_auto

    def __rrshift__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_rshift_, (other, self)))
        return new_auto

    def __mod__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_mod_, (self, other)))
        return new_auto

    def __rmod__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_mod_, (other, self)))
        return new_auto

    def __mul__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_mul_, (self, other)))
        return new_auto

    def __rmul__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_mul_, (other, self)))
        return new_auto

    def __pow__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_pow_, (self, other)))
        return new_auto

    def __rpow__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_pow_, (other, self)))
        return new_auto

    def __sub__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_sub_, (self, other)))
        return new_auto

    def __rsub__(self, other):
        new_auto = self.__class__()
        new_auto._operations = self._operations[:]
        new_auto._operations.append((_sub_, (other, self)))
        return new_auto

    def __repr__(self):
        if self._operations:
            return 'auto(...)'
        else:
            return 'auto(%r, *%r, **%r)' % (self._value, self._args, self._kwds)

    @_bltin_property
    def value(self):
        if self._value is not _auto_null and self._operations:
            raise TypeError('auto() object out of sync')
        elif self._value is _auto_null and not self._operations:
            return self._value
        elif self._value is not _auto_null:
            return self._value
        else:
            return self._resolve()

    @value.setter
    def value(self, value):
        if self._operations:
            value = self._resolve(value)
        self._value = value

    def _resolve(self, base_value=None):
            cls = self.__class__
            for op, params in self._operations:
                values = []
                for param in params:
                    if isinstance(param, cls):
                        if param.value is _auto_null:
                            if base_value is None:
                                return _auto_null
                            else:
                                values.append(base_value)
                        else:
                            values.append(param.value)
                    else:
                        values.append(param)
                value = op(*values)
            self._operations[:] = []
            self._value = value
            return value


class _EnumArgSpec(NamedTuple):
    args = 0, 'all args except *args and **kwds'
    varargs = 1, 'the name of the *args variable'
    keywords = 2, 'the name of the **kwds variable'
    defaults = 3, 'any default values'
    required = 4, 'number of required values (no default available)'

    def __new__(cls, _new_func):
        argspec = getargspec(_new_func)
        args, varargs, keywords, defaults = argspec
        if defaults:
            reqs = args[1:-len(defaults)]
        else:
            reqs = args[1:]
        return tuple.__new__(_EnumArgSpec, (args, varargs, keywords, defaults, reqs))


class _proto_member:
    """
    intermediate step for enum members between class execution and final creation
    """

    def __init__(self, value):
        self.value = value

    def __set_name__(self, enum_class, member_name):
        """
        convert each quasi-member into an instance of the new enum class
        """
        # first step: remove ourself from enum_class
        delattr(enum_class, member_name)
        # second step: create member based on enum_class
        value = self.value
        kwds = {}
        args = ()
        init_args = ()
        extra_mv_args = ()
        multivalue = None
        if isinstance(value, tuple) and value and isinstance(value[0], auto):
            multivalue = value
            value = value[0]
        if isinstance(value, auto) and value.value is _auto_null:
            args = value.args
            kwds = value.kwds
        elif isinstance(value, auto):
            kwds = value.kwds
            args = (value.value, ) + value.args
            value = value.value
        elif isinstance(value, enum):
            args = value.args
            kwds = value.kwds
        elif isinstance(value, Member):
            value = value.value
            args = (value, )
        elif not isinstance(value, tuple):
            args = (value, )
        else:
            args = value
        if multivalue is not None:
            value = (value, ) + multivalue[1:]
            kwds = {}
            args = value
            del multivalue
        # possibilities
        #
        # - no init, multivalue  -> __new__[0], __init__(*[:]), extra=[1:]
        # - init w/o value, multivalue  -> __new__[0], __init__(*[:]), extra=[1:]
        #
        # - init w/value, multivalue  -> __new__[0], __init__(*[1:]),  extra=[1:]
        #
        # - init w/value, no multivalue  -> __new__[0], __init__(*[1:]), extra=[]
        #
        # - init w/o value, no multivalue  -> __new__[:], __init__(*[:]), extra=[]
        # - no init, no multivalue  ->  __new__[:], __init__(*[:]), extra=[]
        if enum_class._multivalue_ or 'value' in enum_class._creating_init_:
            if enum_class._multivalue_:
                # when multivalue is True, creating_init can be anything
                mv_arg = args[0]
                extra_mv_args = args[1:]
                if 'value' in enum_class._creating_init_:
                    init_args = args[1:]
                else:
                    init_args = args
                args = args[0:1]
                value = args[0]
            else:
                # 'value' is definitely in creating_init
                if enum_class._auto_init_ and enum_class._new_args_:
                    # we have a custom __new__ and an auto __init__
                    # divvy up according to number of params in each
                    init_args = args[-len(enum_class._creating_init_)+1:]
                    if not enum_class._auto_args_:
                        args = args[:len(enum_class._new_args_.args)]
                    value = args[0]
                elif enum_class._auto_init_:
                    # don't pass in value
                    init_args = args[1:]
                    args = args[0:1]
                    value = args[0]
                elif enum_class._new_args_:
                    # do not modify args
                    value = args[0]
                else:
                    # keep all args for user-defined __init__
                    # keep value as-is
                    init_args = args
        else:
            # either no creating_init, or it doesn't have 'value'
            init_args = args
        if enum_class._member_type_ is tuple:   # special case for tuple enums
            args = (args, )     # wrap it one more time
        if not enum_class._use_args_:
            enum_member = enum_class._new_member_(enum_class)
            if not hasattr(enum_member, '_value_'):
                enum_member._value_ = value
        else:
            enum_member = enum_class._new_member_(enum_class, *args, **kwds)
            if not hasattr(enum_member, '_value_'):
                if enum_class._member_type_ is object:
                    enum_member._value_ = value
                else:
                    try:
                        enum_member._value_ = enum_class._member_type_(*args, **kwds)
                    except Exception:
                        te = TypeError('_value_ not set in __new__, unable to create it')
                        te.__cause__ = None
                        raise te
        value = enum_member._value_
        enum_member._name_ = member_name
        enum_member.__objclass__ = enum_class
        enum_member.__init__(*init_args, **kwds)
        enum_member._sort_order_ = len(enum_class._member_names_)
        # If another member with the same value was already defined, the
        # new member becomes an alias to the existing one.
        if enum_class._noalias_:
            # unless NoAlias was specified
            enum_class._member_names_.append(member_name)
        else:
            nonunique = defaultdict(list)
            try:
                try:
                    # try to do a fast lookup to avoid the quadratic loop
                    enum_member = enum_class._value2member_map_[value]
                    if enum_class._unique_:
                        nonunique[enum_member.name].append(member_name)
                except TypeError:
                    # unhashable members are stored elsewhere
                    for unhashable_value, canonical_member in enum_class._value2member_seq_:
                        name = canonical_member.name
                        if unhashable_value == enum_member._value_:
                            if enum_class._unique_:
                                nonunique[name].append(member_name)
                            enum_member = canonical_member
                            break
                    else:
                        raise KeyError
            except KeyError:
                # this could still be an alias if the value is multi-bit and the
                # class is a flag class
                if (
                        Flag is None
                        or not issubclass(enum_class, Flag)
                    ):
                    # no other instances found, record this member in _member_names_
                    enum_class._member_names_.append(member_name)
                elif (
                        Flag is not None
                        and issubclass(enum_class, Flag)
                        and _is_single_bit(value)
                    ):
                    # no other instances found, record this member in _member_names_
                    enum_class._member_names_.append(member_name)
            if nonunique:
                # duplicates not allowed if Unique specified
                message = []
                for name, aliases in nonunique.items():
                    bad_aliases = ','.join(aliases)
                    message.append('%s --> %s [%r]' % (name, bad_aliases, enum_class[name].value))
                raise ValueError(
                        '%s: duplicate names found: %s' %
                            (enum_class.__name__, ';  '.join(message))
                        )
        # if self.value is an `auto()`, replace the value attribute with the new enum member
        if isinstance(self.value, auto):
            self.value.enum_member = enum_member
        # get redirect in place before adding to _member_map_
        # but check for other instances in parent classes first
        need_override = False
        descriptor = None
        descriptor_property = None
        for base in enum_class.__mro__[1:]:
            descriptor = base.__dict__.get(member_name)
            if descriptor is not None:
                if isinstance(descriptor, (property, DynamicClassAttribute)):
                    break
                else:
                    need_override = True
                    if isinstance(descriptor, _bltin_property) and descriptor_property is None:
                        descriptor_property = descriptor
                    # keep looking for an enum.property
        descriptor = descriptor or descriptor_property
        if descriptor and not need_override:
            # previous enum.property found, no further action needed
            pass
        else:
            redirect = property()
            redirect.__set_name__(enum_class, member_name)
            if descriptor and need_override:
                # previous enum.property found, but some other inherited
                # attribute is in the way; copy fget, fset, fdel to this one
                redirect.fget = descriptor.fget
                redirect.fset = descriptor.fset
                redirect.fdel = descriptor.fdel
            setattr(enum_class, member_name, redirect)
        # now add to _member_map_ (even aliases)
        enum_class._member_map_[member_name] = enum_member
        #
        # process (possible) MultiValues
        values = (value, ) + extra_mv_args
        if enum_class._multivalue_ and mv_arg not in values:
            values += (mv_arg, )
        enum_member._values_ = values
        for value in values:
            # first check if value has already been used
            if enum_class._multivalue_ and (
                    value in enum_class._value2member_map_
                    or any(v == value for (v, m) in enum_class._value2member_seq_)
                    ):
                raise ValueError('%r has already been used' % (value, ))
            try:
                # This may fail if value is not hashable. We can't add the value
                # to the map, and by-value lookups for this value will be
                # linear.
                if enum_class._noalias_:
                    raise TypeError('cannot use dict to store value')
                enum_class._value2member_map_[value] = enum_member
            except TypeError:
                enum_class._value2member_seq_ += ((value, enum_member), )


class _EnumDict(dict):
    """Track enum member order and ensure member names are not reused.

    EnumType will use the names found in self._member_names as the
    enumeration member names.
    """
    def __init__(self, cls_name, settings, start, constructor_init, constructor_start, constructor_boundary):
        super(_EnumDict, self).__init__()
        self._cls_name = cls_name
        self._constructor_init = constructor_init
        self._constructor_start = constructor_start
        self._constructor_boundary = constructor_boundary
        self._generate_next_value = None
        self._member_names = []
        self._member_names_set = set()
        self._settings = settings
        self._addvalue = addvalue = AddValue in settings
        self._magicvalue = MagicValue in settings
        self._multivalue = MultiValue in settings
        if self._addvalue and self._magicvalue:
            raise TypeError('%r: AddValue and MagicValue are mutually exclusive' % cls_name)
        if self._multivalue and self._magicvalue:
            raise TypeError('%r: MultiValue and MagicValue are mutually exclusive' % cls_name)
        self._start = start
        self._addvalue_value = start
        self._new_args = ()
        self._auto_args = False
        # when the magic turns off
        self._locked = MagicValue not in settings
        # if init fields are specified
        self._init = []
        # list of temporary names
        self._ignore = []
        if self._magicvalue:
            self._ignore = ['property', 'staticmethod', 'classmethod']
        self._ignore_init_done = False
        # if _sunder_ values can be changed via the class body
        self._allow_init = True
        self._last_values = []

    def __getitem__(self, key):
        if key == self._cls_name and self._cls_name not in self:
            return enum
        elif (
                self._locked
                or key in self
                or key in self._ignore
                or _is_sunder(key)
                or _is_dunder(key)
                ):
            return super(_EnumDict, self).__getitem__(key)
        elif self._magicvalue:
            value = self._generate_next_value(key, self._start, len(self._member_names), self._last_values[:])
            self.__setitem__(key, value)
            return value
        else:
            raise Exception('Magic is not set -- why am I here?')

    def __setitem__(self, key, value):
        """Changes anything not sundured, dundered, nor a descriptor.

        If an enum member name is used twice, an error is raised; duplicate
        values are not checked for.

        Single underscore (sunder) names are reserved.
        """
        # Flag classes that have MagicValue and __new__ will get a generated _gnv_
        if _is_internal_class(self._cls_name, value):
            pass
        elif _is_private_name(self._cls_name, key):
            pass
        elif _is_sunder(key):
            if key not in (
                    '_init_', '_settings_', '_order_', '_ignore_', '_start_',
                    '_create_pseudo_member_', '_create_pseudo_member_values_',
                    '_generate_next_value_', '_boundary_', '_numeric_repr_',
                    '_missing_', '_missing_value_', '_missing_name_',
                    '_iter_member_', '_iter_member_by_value_', '_iter_member_by_def_',
                    ):
                raise ValueError('%r: _sunder_ names, such as %r, are reserved for future Enum use'
                        % (self._cls_name, key)
                        )
            elif not self._allow_init and key not in (
                    'create_pseudo_member_', '_missing_', '_missing_value_', '_missing_name_',
                ):
                # sunder is used during creation, must be specified first
                raise ValueError('%r: cannot set %r after init phase' % (self._cls_name, key))
            elif key == '_ignore_':
                if self._ignore_init_done:
                    raise TypeError('%r: ignore can only be specified once' % self._cls_name)
                if isinstance(value, basestring):
                    value = value.split()
                else:
                    value = list(value)
                self._ignore = value
                already = set(value) & self._member_names_set
                if already:
                    raise ValueError('%r: _ignore_ cannot specify already set names %s' % (
                            self._cls_name,
                            ', '.join(repr(a) for a in already)
                            ))
                self._ignore_init_done = True
            elif key == '_boundary_':
                if self._constructor_boundary:
                    raise TypeError('%r: boundary specified in constructor and class body' % self._cls_name)
            elif key == '_start_':
                if self._constructor_start:
                    raise TypeError('%r: start specified in constructor and class body' % self._cls_name)
                self._start = value
            elif key == '_settings_':
                if not isinstance(value, (set, tuple)):
                    value = (value, )
                if not isinstance(value, set):
                    value = set(value)
                self._settings |= value
                if NoAlias in value and Unique in value:
                    raise TypeError('%r: NoAlias and Unique are mutually exclusive' % self._cls_name)
                elif MultiValue in value and NoAlias in value:
                    raise TypeError('cannot specify both MultiValue and NoAlias' % self._cls_name)
                allowed_settings = dict.fromkeys(['addvalue', 'magicvalue', 'noalias', 'unique', 'multivalue'])
                for arg in self._settings:
                    if arg not in allowed_settings:
                        raise TypeError('%r: unknown qualifier %r (from %r)' % (self._cls_name, arg, value))
                    allowed_settings[arg] = True
                self._multivalue = allowed_settings['multivalue']
                self._addvalue = allowed_settings['addvalue']
                self._magicvalue = allowed_settings['magicvalue']
                self._locked = not self._magicvalue
                if self._magicvalue and not self._ignore_init_done:
                    self._ignore = ['property', 'classmethod', 'staticmethod']
                if self._addvalue and self._init and 'value' not in self._init:
                    self._init.insert(0, 'value')
                value = tuple(self._settings)
            elif key == '_init_':
                if self._constructor_init:
                    raise TypeError('%r: init specified in constructor and in class body' % self._cls_name)
                _init_ = value
                if isinstance(_init_, basestring):
                    _init_ = _init_.replace(',',' ').split()
                if self._addvalue and 'value' not in self._init:
                    self._init.insert(0, 'value')
                if self._magicvalue:
                    raise TypeError("%r: _init_ and MagicValue are mutually exclusive" % self._cls_name)
                self._init = _init_
                value = _init_
            elif key == '_generate_next_value_':
                gnv = value
                if value is not None:
                    if isinstance(value, staticmethod):
                        gnv = value.__func__
                    elif isinstance(value, classmethod):
                        raise TypeError('%r: _generate_next_value must be a staticmethod, not a classmethod' % self._cls_name)
                    else:
                        gnv = value
                        value = staticmethod(value)
                    self._auto_args = _check_auto_args(value)
                setattr(self, '_generate_next_value', gnv)
        elif _is_dunder(key):
            if key == '__order__':
                key = '_order_'
                if not self._allow_init:
                    # _order_ is used during creation, must be specified first
                    raise ValueError('%r: cannot set %r after init phase' % (self._cls_name, key))
            elif key == '__new__':  # and self._new_to_init:
                if isinstance(value, staticmethod):
                    value = value.__func__
                self._new_args = _EnumArgSpec(value)
            elif key == '__init_subclass__':
                if not isinstance(value, classmethod):
                    value = classmethod(value)
            if _is_descriptor(value):
                self._locked = True
        elif key in self._member_names_set:
            # descriptor overwriting an enum?
            raise TypeError('%r: attempt to reuse name: %r' % (self._cls_name, key))
        elif key in self._ignore:
            pass
        elif not _is_descriptor(value):
            self._allow_init = False
            if key in self:
                # enum overwriting a descriptor?
                raise TypeError('%r: %s already defined as %r' % (self._cls_name, key, self[key]))
            if type(value) is enum:
                value.name = key
                if self._addvalue:
                    raise TypeError('%r: enum() and AddValue are incompatible' % self._cls_name)
            elif self._addvalue and not self._multivalue:
                # generate a value
                value = self._gnv(key, value)
            elif self._multivalue:
                # make sure it's a tuple
                if not isinstance(value, tuple):
                    value = (value, )
                if isinstance(value[0], auto):
                    value = (self._convert_auto(key, value[0]), ) + value[1:]
                if self._addvalue:
                    value = self._gnv(key, value)
            elif isinstance(value, auto):
                value = self._convert_auto(key, value)
            elif isinstance(value, tuple) and value and isinstance(value[0], auto):
                value = (self._convert_auto(key, value[0]), ) + value[1:]
            elif not isinstance(value, auto):
                # call generate maybe if
                # - init is specified; or
                # - __new__ is specified;
                # and either of them call for more values than are present
                new_args = () or self._new_args and self._new_args.required
                target_len = len(self._init or new_args)
                if isinstance(value, tuple):
                    source_len = len(value)
                else:
                    source_len = 1
                multi_args = len(self._init) > 1 or new_args
                if source_len < target_len :
                    value = self._gnv(key, value)
            else:
                pass
            if self._init:
                if isinstance(value, auto):
                    test_value = value.args
                elif not isinstance(value, tuple):
                    test_value = (value, )
                else:
                    test_value = value
                if len(self._init) != len(test_value):
                    raise TypeError(
                            '%s.%s: number of fields provided do not match init [%r != %r]'
                            % (self._cls_name, key, self._init, test_value)
                        )
            self._member_names.append(key)
            self._member_names_set.add(key)
        else:
            # not a new member, turn off the autoassign magic
            self._locked = True
            self._allow_init = False
        if not (_is_sunder(key) or _is_dunder(key) or _is_private_name(self._cls_name, key) or _is_descriptor(value)):
            if isinstance(value, auto):
                self._last_values.append(value.value)
            elif isinstance(value, tuple) and value and isinstance(value[0], auto):
                self._last_values.append(value[0].value)
            elif isinstance(value, tuple):
                if value:
                    self._last_values.append(value[0])
            else:
                self._last_values.append(value)
        super(_EnumDict, self).__setitem__(key, value)

    def _convert_auto(self, key, value):
        # if auto.args or auto.kwds, compare to _init_ and __new__ -- if lacking, call gnv
        # if not auto.args|kwds but auto.value is _auto_null -- call gnv
        if value.args or value.kwds or value.value is _auto_null:
            if value.args or value.kwds:
                values = value.args
            else:
                values = ()
            new_args = () or self._new_args and self._new_args.required
            target_len = len(self._init or new_args) or 1
            if isinstance(values, tuple):
                source_len = len(values)
            else:
                source_len = 1
            multi_args = len(self._init) > 1 or new_args
            if source_len < target_len :
                values = self._gnv(key, values)
                if value.args:
                    value._args = values
                else:
                    value.value = values
        return value

    def _gnv(self, key, value):
        # generate a value
        if self._auto_args:
            if not isinstance(value, tuple):
                value = (value, )
            value = self._generate_next_value(key, self._start, len(self._member_names), self._last_values[:], *value)
        else:
            value = self._generate_next_value(key, self._start, len(self._member_names), self._last_values[:])
        if isinstance(value, tuple) and len(value) == 1:
            value = value[0]
        return value


no_arg = SentinelType('no_arg', (object, ), {})
class EnumType(type):
    """Metaclass for Enum"""

    @classmethod
    def __prepare__(metacls, cls, bases, init=None, start=None, settings=(), boundary=None, **kwds):
        metacls._check_for_existing_members_(cls, bases)
        if Flag is None and cls == 'Flag':
            initial_flag = True
        else:
            initial_flag = False
        # settings are a combination of current and all past settings
        constructor_init = init is not None
        constructor_start = start is not None
        constructor_boundary = boundary is not None
        if not isinstance(settings, tuple):
            settings = settings,
        settings = set(settings)
        generate = None
        order = None
        # inherit previous flags
        member_type, first_enum = metacls._get_mixins_(cls, bases)
        if first_enum is not None:
            generate = getattr(first_enum, '_generate_next_value_', None)
            generate = getattr(generate, 'im_func', generate)
            settings |= metacls._get_settings_(bases)
            init = init or first_enum._auto_init_[:]
            order = first_enum._order_function_
            if start is None:
                start = first_enum._start_
        else:
            # first time through -- creating Enum itself
            start = 1
        # check for custom settings
        if AddValue in settings and init and 'value' not in init:
            if isinstance(init, list):
                init.insert(0, 'value')
            else:
                init = 'value ' + init
        if NoAlias in settings and Unique in settings:
            raise TypeError('%r: NoAlias and Unique are mutually exclusive' % cls)
        if MultiValue in settings and NoAlias in settings:
            raise TypeError('%r: MultiValue and NoAlias are mutually exclusive' % cls)
        allowed_settings = dict.fromkeys(['addvalue', 'magicvalue', 'noalias', 'unique', 'multivalue'])
        for arg in settings:
            if arg not in allowed_settings:
                raise TypeError('%r: unknown qualifier %r' % (cls, arg))
        enum_dict = _EnumDict(cls_name=cls, settings=settings, start=start, constructor_init=constructor_init, constructor_start=constructor_start, constructor_boundary=constructor_boundary)
        enum_dict._member_type = member_type
        enum_dict._base_type = ('enum', 'flag')[
                Flag is None and cls == 'Flag'
                or
                Flag is not None and any(issubclass(b, Flag) for b in bases)
                ]
        if Flag is not None and any(b is Flag for b in bases) and member_type not in (baseinteger + (object, )):
            if Flag in bases:
                # when a non-int data type is mixed in with Flag, we end up
                # needing two values for two `__new__`s:
                # - the integer value for the Flag itself; and
                # - the mix-in value for the mix-in
                #
                # we provide a default `_generate_next_value_` to supply the int
                # argument, and a default `__new__` to keep the two straight
                def _generate_next_value_(name, start, count, values, *args, **kwds):
                    return (2 ** count, ) + args
                enum_dict['_generate_next_value_'] = staticmethod(_generate_next_value_)
                def __new__(cls, flag_value, type_value):
                    obj = member_type.__new__(cls, type_value)
                    obj._value_ = flag_value
                    return obj
                enum_dict['__new__'] = __new__
            else:
                try:
                    enum_dict._new_args = _EnumArgSpec(first_enum.__new_member__)
                except TypeError:
                    pass
        elif not initial_flag:
            if hasattr(first_enum, '__new_member__'):
                enum_dict._new_args = _EnumArgSpec(first_enum.__new_member__)
            if generate:
                enum_dict['_generate_next_value_'] = generate
                enum_dict._inherited_gnv = True
            if init is not None:
                if isinstance(init, basestring):
                    init = init.replace(',',' ').split()
                enum_dict._init = init
        elif hasattr(first_enum, '__new_member__'):
            enum_dict._new_args = _EnumArgSpec(first_enum.__new_member__)
        if order is not None:
            enum_dict['_order_'] = staticmethod(order)
        return enum_dict

    def __init__(cls, *args , **kwds):
        pass

    def __new__(metacls, cls, bases, clsdict, init=None, start=None, settings=(), boundary=None, **kwds):
        # handle py2 case first
        if type(clsdict) is not _EnumDict:
            # py2 and/or functional API gyrations
            init = clsdict.pop('_init_', None)
            start = clsdict.pop('_start_', None)
            settings = clsdict.pop('_settings_', ())
            _order_ = clsdict.pop('_order_', clsdict.pop('__order__', None))
            _ignore_ = clsdict.pop('_ignore_', None)
            _create_pseudo_member_ = clsdict.pop('_create_pseudo_member_', None)
            _create_pseudo_member_values_ = clsdict.pop('_create_pseudo_member_values_', None)
            _generate_next_value_ = clsdict.pop('_generate_next_value_', None)
            _missing_ = clsdict.pop('_missing_', None)
            _missing_value_ = clsdict.pop('_missing_value_', None)
            _missing_name_ = clsdict.pop('_missing_name_', None)
            _boundary_ = clsdict.pop('_boundary_', None)
            _iter_member_ = clsdict.pop('_iter_member_', None)
            _iter_member_by_value_ = clsdict.pop('_iter_member_by_value_', None)
            _iter_member_by_def_ = clsdict.pop('_iter_member_by_def_', None)
            __new__ = clsdict.pop('__new__', None)
            __new__ = getattr(__new__, 'im_func', __new__)
            __new__ = getattr(__new__, '__func__', __new__)
            enum_members = dict([
                    (k, v) for (k, v) in clsdict.items()
                    if not (_is_sunder(k) or _is_dunder(k) or _is_private_name(cls, k) or _is_descriptor(v))
                    ])
            original_dict = clsdict
            clsdict = metacls.__prepare__(cls, bases, init=init, start=start)
            if settings:
                clsdict['_settings_'] = settings
            init = init or clsdict._init
            if _order_ is None:
                _order_ = clsdict.get('_order_')
                if _order_ is not None:
                    _order_ = _order_.__get__(cls)
            if isinstance(original_dict, OrderedDict):
                calced_order = original_dict
            elif _order_ is None:
                calced_order = [name for (name, value) in enumsort(list(enum_members.items()))]
            elif isinstance(_order_, basestring):
                calced_order = _order_ = _order_.replace(',', ' ').split()
            elif callable(_order_):
                if init:
                    if not isinstance(init, basestring):
                        init = ' '.join(init)
                member = NamedTuple('member', init and 'name ' + init or ['name', 'value'])
                calced_order = []
                for name, value in enum_members.items():
                    if init:
                        if not isinstance(value, tuple):
                            value = (value, )
                        name_value = (name, ) + value
                    else:
                        name_value = tuple((name, value))
                    if member._defined_len_ != len(name_value):
                        raise TypeError('%d values expected (%s), %d received (%s)' % (
                            member._defined_len_,
                            ', '.join(member._fields_),
                            len(name_value),
                            ', '.join([repr(v) for v in name_value]),
                            ))
                    calced_order.append(member(*name_value))
                calced_order = [m.name for m in sorted(calced_order, key=_order_)]
            else:
                calced_order = _order_
            for name in (
                    '_missing_', '_missing_value_', '_missing_name_',
                    '_ignore_', '_create_pseudo_member_', '_create_pseudo_member_values_',
                    '_generate_next_value_', '_order_', '__new__',
                    '_missing_', '_missing_value_', '_missing_name_',
                    '_boundary_',
                    '_iter_member_', '_iter_member_by_value_', '_iter_member_by_def_',
                ):
                attr = locals()[name]
                if attr is not None:
                    clsdict[name] = attr
            # now add members
            for k in calced_order:
                try:
                    clsdict[k] = original_dict[k]
                except KeyError:
                    # this error will be handled when _order_ is checked
                    pass
            for k, v in original_dict.items():
                if k not in calced_order:
                    clsdict[k] = v
            del _order_, _ignore_, _create_pseudo_member_, _create_pseudo_member_values_,
            del _generate_next_value_, _missing_, _missing_value_, _missing_name_
        #
        # resume normal path
        clsdict._locked = True
        #
        # check for illegal enum names (any others?)
        member_names = clsdict._member_names
        invalid_names = set(member_names) & set(['mro', ''])
        if invalid_names:
            raise ValueError('invalid enum member name(s): %s' % (
                ', '.join(invalid_names), ))
        _order_ = clsdict.pop('_order_', None)
        if isinstance(_order_, basestring):
            _order_ = _order_.replace(',',' ').split()
        init = clsdict._init
        start = clsdict._start
        settings = clsdict._settings
        creating_init = []
        new_args = clsdict._new_args
        auto_args = clsdict._auto_args
        auto_init = False
        if init is not None:
            auto_init = True
            creating_init = init[:]
        if 'value' in creating_init and creating_init[0] != 'value':
            raise TypeError("'value', if specified, must be the first item in 'init'")
        magicvalue = MagicValue in settings
        multivalue = MultiValue in settings
        noalias = NoAlias in settings
        unique = Unique in settings
        # an Enum class cannot be mixed with other types (int, float, etc.) if
        #   it has an inherited __new__ unless a new __new__ is defined (or
        #   the resulting class will fail).
        # an Enum class is final once enumeration items have been defined;
        #
        # remove any keys listed in _ignore_
        clsdict.setdefault('_ignore_', []).append('_ignore_')
        ignore = clsdict['_ignore_']
        for key in ignore:
            clsdict.pop(key, None)
        #
        boundary = boundary or clsdict.pop('_boundary_', None)
        # convert to regular dict
        clsdict = dict(clsdict.items())
        member_type, first_enum = metacls._get_mixins_(cls, bases)
        # get the method to create enum members
        __new__, save_new, new_uses_args = metacls._find_new_(
                clsdict,
                member_type,
                first_enum,
                )
        clsdict['_new_member_'] = staticmethod(__new__)
        clsdict['_use_args_'] = new_uses_args
        #
        # convert future enum members into temporary _proto_members
        # and record integer values in case this will be a Flag
        flag_mask = 0
        for name in member_names:
            value = test_value = clsdict[name]
            if isinstance(value, auto) and value.value is not _auto_null:
                test_value = value.value
            if isinstance(test_value, baseinteger):
                flag_mask |= test_value
            if isinstance(test_value, tuple) and test_value and isinstance(test_value[0], baseinteger):
                flag_mask |= test_value[0]
            clsdict[name] = _proto_member(value)
        #
        # temp stuff
        clsdict['_creating_init_'] = creating_init
        clsdict['_multivalue_'] = multivalue
        clsdict['_magicvalue_'] = magicvalue
        clsdict['_noalias_'] = noalias
        clsdict['_unique_'] = unique
        #
        # house-keeping structures
        clsdict['_member_names_'] = []
        clsdict['_member_map_'] = OrderedDict()
        clsdict['_member_type_'] = member_type
        clsdict['_value2member_map_'] = {}
        clsdict['_value2member_seq_'] = ()
        clsdict['_settings_'] = settings
        clsdict['_start_'] = start
        clsdict['_auto_init_'] = init
        clsdict['_new_args_'] = new_args
        clsdict['_auto_args_'] = auto_args
        clsdict['_order_function_'] = None
        # now set the __repr__ for the value
        clsdict['_value_repr_'] = metacls._find_data_repr_(cls, bases)
        #
        # Flag structures (will be removed if final class is not a Flag
        clsdict['_boundary_'] = (
                boundary
                or getattr(first_enum, '_boundary_', None)
                )
        clsdict['_flag_mask_'] = flag_mask
        clsdict['_all_bits_'] = 2 ** ((flag_mask).bit_length()) - 1
        clsdict['_inverted_'] = None
        #
        # move skipped values out of the descriptor
        for name, obj in clsdict.items():
            if isinstance(obj, nonmember):
                clsdict[name] = obj.value
        #
        # If a custom type is mixed into the Enum, and it does not know how
        # to pickle itself, pickle.dumps will succeed but pickle.loads will
        # fail.  Rather than have the error show up later and possibly far
        # from the source, sabotage the pickle protocol for this class so
        # that pickle.dumps also fails.
        #
        # However, if the new class implements its own __reduce_ex__, do not
        # sabotage -- it's on them to make sure it works correctly.  We use
        # __reduce_ex__ instead of any of the others as it is preferred by
        # pickle over __reduce__, and it handles all pickle protocols.
        unpicklable = False
        if '__reduce_ex__' not in clsdict:
            if member_type is not object:
                methods = ('__getnewargs_ex__', '__getnewargs__',
                        '__reduce_ex__', '__reduce__')
                if not any(m in member_type.__dict__ for m in methods):
                    _make_class_unpicklable(clsdict)
                    unpicklable = True
        #
        # create a default docstring if one has not been provided
        if '__doc__' not in clsdict:
            clsdict['__doc__'] = 'An enumeration.'
        #
        # create our new Enum type
        try:
            exc = None
            enum_class = type.__new__(metacls, cls, bases, clsdict)
        except RuntimeError as e:
            # any exceptions raised by _proto_member (aka member.__new__) will get converted to
            # a RuntimeError, so get that original exception back and raise
            # it instead
            exc = e.__cause__ or e
        if exc is not None:
            raise exc
        #
        # if Python 3.5 or ealier, implement the __set_name__ and
        # __init_subclass__ protocols
        if pyver < PY3_6:
            for name in member_names:
                enum_class.__dict__[name].__set_name__(enum_class, name)
            for name, obj in enum_class.__dict__.items():
                if name in member_names:
                    continue
                if hasattr(obj, '__set_name__'):
                    obj.__set_name__(enum_class, name)
            if Enum is not None:
                super(enum_class, enum_class).__init_subclass__()
        #
        # double check that repr and friends are not the mixin's or various
        # things break (such as pickle)
        #
        # Also, special handling for ReprEnum
        if ReprEnum is not None and ReprEnum in bases:
            if member_type is object:
                raise TypeError(
                        'ReprEnum subclasses must be mixed with a data type (i.e.'
                        ' int, str, float, etc.)'
                        )
            if '__format__' not in clsdict:
                enum_class.__format__ = member_type.__format__
                clsdict['__format__'] = enum_class.__format__
            if '__str__' not in clsdict:
                method = member_type.__str__
                if method is object.__str__:
                    # if member_type does not define __str__, object.__str__ will use
                    # its __repr__ instead, so we'll also use its __repr__
                    method = member_type.__repr__
                enum_class.__str__ = method
                clsdict['__str__'] = enum_class.__str__

        for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
            if name in clsdict:
                # class has defined/imported/copied the method
                continue
            class_method = getattr(enum_class, name)
            obj_method = getattr(member_type, name, None)
            enum_method = getattr(first_enum, name, None)
            if obj_method is not None and obj_method == class_method:
                if name == '__reduce_ex__' and unpicklable:
                    continue
                setattr(enum_class, name, enum_method)
                clsdict[name] = enum_method
        #
        # for Flag, add __or__, __and__, __xor__, and __invert__
        if Flag is not None and issubclass(enum_class, Flag):
            for name in (
                    '__or__', '__and__', '__xor__',
                    '__ror__', '__rand__', '__rxor__',
                    '__invert__'
                ):
                if name not in clsdict:
                    setattr(enum_class, name, getattr(Flag, name))
                    clsdict[name] = enum_method
        #
        # method resolution and int's are not playing nice
        # Python's less than 2.6 use __cmp__
        if pyver < PY2_6:
            #
            if issubclass(enum_class, int):
                setattr(enum_class, '__cmp__', getattr(int, '__cmp__'))
            #
        elif PY2:
            #
            if issubclass(enum_class, int):
                for method in (
                        '__le__',
                        '__lt__',
                        '__gt__',
                        '__ge__',
                        '__eq__',
                        '__ne__',
                        '__hash__',
                        ):
                    setattr(enum_class, method, getattr(int, method))
        #
        # replace any other __new__ with our own (as long as Enum is not None,
        # anyway) -- again, this is to support pickle
        if Enum is not None:
            # if the user defined their own __new__, save it before it gets
            # clobbered in case they subclass later
            if save_new:
                setattr(enum_class, '__new_member__', enum_class.__dict__['__new__'])
            setattr(enum_class, '__new__', Enum.__dict__['__new__'])
        #
        # _order_ checking is spread out into three/four steps
        # - ensure _order_ is a list, not a string nor a function
        # - if enum_class is a Flag:
        #   - remove any non-single-bit flags from _order_
        # - remove any aliases from _order_
        # - check that _order_ and _member_names_ match
        #
        # _order_ step 1: ensure _order_ is a list
        if _order_:
            if isinstance(_order_, staticmethod):
                _order_ = _order_.__func__
            if callable(_order_):
                # save order for future subclasses
                enum_class._order_function_ = staticmethod(_order_)
                # create ordered list for comparison
                _order_ = [m.name for m in sorted(enum_class, key=_order_)]
        #
        # remove Flag structures if final class is not a Flag
        if (
                Flag is None and cls != 'Flag'
                or Flag is not None and not issubclass(enum_class, Flag)
            ):
            delattr(enum_class, '_boundary_')
            delattr(enum_class, '_flag_mask_')
            delattr(enum_class, '_all_bits_')
            delattr(enum_class, '_inverted_')
        elif Flag is not None and issubclass(enum_class, Flag):
            # ensure _all_bits_ is correct and there are no missing flags
            single_bit_total = 0
            multi_bit_total = 0
            for flag in enum_class._member_map_.values():
                if _is_single_bit(flag._value_):
                    single_bit_total |= flag._value_
                else:
                    # multi-bit flags are considered aliases
                    multi_bit_total |= flag._value_
            if enum_class._boundary_ is not KEEP:
                missed = list(_iter_bits_lsb(multi_bit_total & ~single_bit_total))
                if missed:
                    raise TypeError(
                            'invalid Flag %r -- missing values: %s'
                            % (cls, ', '.join((str(i) for i in missed)))
                            )
            enum_class._flag_mask_ = single_bit_total
            enum_class._all_bits_ = 2 ** ((single_bit_total).bit_length()) - 1
            #
            # set correct __iter__
            if [m._value_ for m in enum_class] != sorted([m._value_ for m in enum_class]):
                enum_class._iter_member_ = enum_class._iter_member_by_def_
            if _order_:
                # _order_ step 2: remove any items from _order_ that are not single-bit
                _order_ = [
                        o
                        for o in _order_
                        if o not in enum_class._member_map_ or _is_single_bit(enum_class[o]._value_)
                        ]
        #
        # check for constants with auto() values
        for k, v in enum_class.__dict__.items():
            if isinstance(v, constant) and isinstance(v.value, auto):
                v.value = enum_class(v.value.value)
        #
        if _order_:
            # _order_ step 3: remove aliases from _order_
            _order_ = [
                    o
                    for o in _order_
                    if (
                        o not in enum_class._member_map_
                        or
                        (o in enum_class._member_map_ and o in enum_class._member_names_)
                        )]
            # _order_ step 4: verify that _order_ and _member_names_ match
            if _order_ != enum_class._member_names_:
                raise TypeError(
                        'member order does not match _order_:\n%r\n%r'
                        % (enum_class._member_names_, _order_)
                        )
        return enum_class

    def __bool__(cls):
        """
        classes/types should always be True.
        """
        return True

    def __call__(cls, value=no_arg, names=None, module=None, qualname=None, type=None, start=1, boundary=None):
        """Either returns an existing member, or creates a new enum class.

        This method is used both when an enum class is given a value to match
        to an enumeration member (i.e. Color(3)) and for the functional API
        (i.e. Color = Enum('Color', names='red green blue')).

        When used for the functional API: `module`, if set, will be stored in
        the new class' __module__ attribute; `type`, if set, will be mixed in
        as the first base class.

        Note: if `module` is not set this routine will attempt to discover the
        calling module by walking the frame stack; if this is unsuccessful
        the resulting class will not be pickleable.
        """
        if names is None:  # simple value lookup
            return cls.__new__(cls, value)
        # otherwise, functional API: we're creating a new Enum type
        return cls._create_(value, names, module=module, qualname=qualname, type=type, start=start, boundary=boundary)

    def __contains__(cls, member):
        if not isinstance(member, Enum):
            raise TypeError("%r (%r) is not an <aenum 'Enum'>" % (member, type(member)))
        if not isinstance(member, cls):
            return False
        return True

    def __delattr__(cls, attr):
        # nicer error message when someone tries to delete an attribute
        # (see issue19025).
        if attr in cls._member_map_:
            raise AttributeError(
                    "%s: cannot delete Enum member %r." % (cls.__name__, attr),
                    )
        found_attr = _get_attr_from_chain(cls, attr)
        if isinstance(found_attr, constant):
            raise AttributeError(
                    "%s: cannot delete constant %r" % (cls.__name__, attr),
                    )
        elif isinstance(found_attr, property):
            raise AttributeError(
                    "%s: cannot delete property %r" % (cls.__name__, attr),
                    )
        super(EnumType, cls).__delattr__(attr)

    def __dir__(cls):
        interesting = set(cls._member_names_ + [
                    '__class__', '__contains__', '__doc__', '__getitem__',
                    '__iter__', '__len__', '__members__', '__module__',
                    '__name__',
                    ])
        if cls._new_member_ is not object.__new__:
            interesting.add('__new__')
        if cls.__init_subclass__ is not Enum.__init_subclass__:
            interesting.add('__init_subclass__')
        if hasattr(object, '__qualname__'):
            interesting.add('__qualname__')
        for method in ('__init__', '__format__', '__repr__', '__str__'):
            if getattr(cls, method) not in (getattr(Enum, method), getattr(Flag, method)):
                interesting.add(method)
        if cls._member_type_ is object:
            return sorted(interesting)
        else:
            # return whatever mixed-in data type has
            return sorted(set(dir(cls._member_type_)) | interesting)

    @_bltin_property
    def __members__(cls):
        """Returns a mapping of member name->value.

        This mapping lists all enum members, including aliases. Note that this
        is a copy of the internal mapping.
        """
        return cls._member_map_.copy()

    def __getitem__(cls, name):
        try:
            return cls._member_map_[name]
        except KeyError:
            exc = _sys.exc_info()[1]
        if Flag is not None and issubclass(cls, Flag) and '|' in name:
            try:
                # may be an __or__ed name
                result = cls(0)
                for n in name.split('|'):
                    result |= cls[n]
                return result
            except KeyError:
                raise exc
        result = cls._missing_name_(name)
        if isinstance(result, cls):
            return result
        else:
            raise exc

    def __iter__(cls):
        return (cls._member_map_[name] for name in cls._member_names_)

    def __reversed__(cls):
        return (cls._member_map_[name] for name in reversed(cls._member_names_))

    def __len__(cls):
        return len(cls._member_names_)

    __nonzero__ = __bool__

    def __repr__(cls):
        return "<aenum %r>" % (cls.__name__, )

    def __setattr__(cls, name, value):
        """Block attempts to reassign Enum members/constants.

        A simple assignment to the class namespace only changes one of the
        several possible ways to get an Enum member from the Enum class,
        resulting in an inconsistent Enumeration.
        """
        member_map = cls.__dict__.get('_member_map_', {})
        if name in member_map:
            raise AttributeError(
                    '%s: cannot rebind member %r.' % (cls.__name__, name),
                    )
        found_attr = _get_attr_from_chain(cls, name)
        if isinstance(found_attr, constant):
            raise AttributeError(
                    "%s: cannot rebind constant %r" % (cls.__name__, name),
                    )
        elif isinstance(found_attr, property):
            raise AttributeError(
                    "%s: cannot rebind property %r" % (cls.__name__, name),
                    )
        super(EnumType, cls).__setattr__(name, value)

    def _convert(cls, *args, **kwds):
        import warnings
        warnings.warn("_convert is deprecated and will be removed, use"
                      " _convert_ instead.", DeprecationWarning, stacklevel=2)
        return cls._convert_(*args, **kwds)

    def _convert_(cls, name, module, filter, source=None, boundary=None, as_global=False):
        """
        Create a new Enum subclass that replaces a collection of global constants
        """
        # convert all constants from source (or module) that pass filter() to
        # a new Enum called name, and export the enum and its members back to
        # module;
        # also, replace the __reduce_ex__ method so unpickling works in
        # previous Python versions
        module_globals = vars(_sys.modules[module])
        if source:
            source = vars(source)
        else:
            source = module_globals
        members = [(key, source[key]) for key in source.keys() if filter(key)]
        try:
            # sort by value, name
            members.sort(key=lambda t: (t[1], t[0]))
        except TypeError:
            # unless some values aren't comparable, in which case sort by just name
            members.sort(key=lambda t: t[0])
        cls = cls(name, members, module=module, boundary=boundary or KEEP)
        cls.__reduce_ex__ = _reduce_ex_by_name
        if as_global:
            global_enum(cls)
        else:
            module_globals.update(cls.__members__)
        module_globals[name] = cls
        return cls

    def _create_(cls, class_name, names, module=None, qualname=None, type=None, start=1, boundary=None):
        """Convenience method to create a new Enum class.

        `names` can be:

        * A string containing member names, separated either with spaces or
          commas.  Values are auto-numbered from 1.
        * An iterable of member names.  Values are auto-numbered from 1.
        * An iterable of (member name, value) pairs.
        * A mapping of member name -> value.
        """
        if PY2:
            # if class_name is unicode, attempt a conversion to ASCII
            if isinstance(class_name, unicode):
                try:
                    class_name = class_name.encode('ascii')
                except UnicodeEncodeError:
                    raise TypeError('%r is not representable in ASCII' % (class_name, ))
        metacls = cls.__class__
        if type is None:
            bases = (cls, )
        else:
            bases = (type, cls)
        _, first_enum = cls._get_mixins_(class_name, bases)
        generate = getattr(first_enum, '_generate_next_value_', None)
        generate = getattr(generate, 'im_func', generate)
        # special processing needed for names?
        if isinstance(names, basestring):
            names = names.replace(',', ' ').split()
        if isinstance(names, (tuple, list)) and names and isinstance(names[0], basestring):
            original_names, names = names, []
            last_values = []
            for count, name in enumerate(original_names):
                value = generate(name, start, count, last_values[:])
                last_values.append(value)
                names.append((name, value))
        # Here, names is either an iterable of (name, value) or a mapping.
        item = None  # in case names is empty
        clsdict = None
        for item in names:
            if clsdict is None:
                # first time initialization
                if isinstance(item, basestring):
                    clsdict = {}
                else:
                    # remember the order
                    clsdict = metacls.__prepare__(class_name, bases)
            if isinstance(item, basestring):
                member_name, member_value = item, names[item]
            else:
                member_name, member_value = item
            clsdict[member_name] = member_value
        if clsdict is None:
            # in case names was empty
            clsdict = metacls.__prepare__(class_name, bases)
        enum_class = metacls.__new__(metacls, class_name, bases, clsdict, boundary=boundary)
        # TODO: replace the frame hack if a blessed way to know the calling
        # module is ever developed
        if module is None:
            try:
                module = _sys._getframe(2).f_globals['__name__']
            except (AttributeError, KeyError):
                pass
        if module is None:
            _make_class_unpicklable(enum_class)
        else:
            enum_class.__module__ = module
        if qualname is not None:
            enum_class.__qualname__ = qualname
        return enum_class

    @classmethod
    def _check_for_existing_members_(mcls, class_name, bases):
        if Enum is None:
            return
        for chain in bases:
            for base in chain.__mro__:
                if issubclass(base, Enum) and base._member_names_:
                    raise TypeError(
                            "<aenum %r> cannot extend %r"
                            % (class_name, base)
                            )
    @classmethod
    def _get_mixins_(mcls, class_name, bases):
        """Returns the type for creating enum members, and the first inherited
        enum class.

        bases: the tuple of bases that was given to __new__
        """
        if not bases or Enum is None:
            return object, Enum

        mcls._check_for_existing_members_(class_name, bases)

        # ensure final parent class is an Enum derivative, find any concrete
        # data type, and check that Enum has no members
        first_enum = bases[-1]
        if not issubclass(first_enum, Enum):
            raise TypeError("new enumerations should be created as "
                    "`EnumName([mixin_type, ...] [data_type,] enum_type)`")
        member_type = mcls._find_data_type_(class_name, bases) or object
        if first_enum._member_names_:
            raise TypeError("cannot extend enumerations via subclassing")
        #
        return member_type, first_enum

    @classmethod
    def _find_data_repr_(mcls, class_name, bases):
        for chain in bases:
            for base in chain.__mro__:
                if base is object:
                    continue
                elif issubclass(base, Enum):
                    # if we hit an Enum, use it's _value_repr_
                    return base._value_repr_
                elif '__repr__' in base.__dict__:
                    # this is our data repr
                    return base.__dict__['__repr__']
        return None

    @classmethod
    def _find_data_type_(mcls, class_name, bases):
        data_types = set()
        for chain in bases:
            candidate = None
            for base in chain.__mro__:
                if base is object or base is StdlibEnum or base is StdlibFlag:
                    continue
                elif issubclass(base, Enum):
                    if base._member_type_ is not object:
                        data_types.add(base._member_type_)
                elif '__new__' in base.__dict__:
                    if issubclass(base, Enum):
                        continue
                    elif StdlibFlag is not None and issubclass(base, StdlibFlag):
                        continue
                    data_types.add(candidate or base)
                    break
                else:
                    candidate = candidate or base
        if len(data_types) > 1:
            raise TypeError('%r: too many data types: %r' % (class_name, data_types))
        elif data_types:
            return data_types.pop()
        else:
            return None

    @staticmethod
    def _get_settings_(bases):
        """Returns the combined _settings_ of all Enum base classes

        bases: the tuple of bases given to __new__
        """
        settings = set()
        for chain in bases:
            for base in chain.__mro__:
                if issubclass(base, Enum):
                    for s in base._settings_:
                        settings.add(s)
        return settings

    @classmethod
    def _find_new_(mcls, clsdict, member_type, first_enum):
        """Returns the __new__ to be used for creating the enum members.

        clsdict: the class dictionary given to __new__
        member_type: the data type whose __new__ will be used by default
        first_enum: enumeration to check for an overriding __new__
        """
        # now find the correct __new__, checking to see of one was defined
        # by the user; also check earlier enum classes in case a __new__ was
        # saved as __new_member__
        __new__ = clsdict.get('__new__', None)
        #
        # should __new__ be saved as __new_member__ later?
        save_new = first_enum is not None and __new__ is not None
        #
        if __new__ is None:
            # check all possibles for __new_member__ before falling back to
            # __new__
            for method in ('__new_member__', '__new__'):
                for possible in (member_type, first_enum):
                    target = getattr(possible, method, None)
                    if target not in (
                            None,
                            None.__new__,
                            object.__new__,
                            Enum.__new__,
                            StdlibEnum.__new__,
                            ):
                        __new__ = target
                        break
                if __new__ is not None:
                    break
            else:
                __new__ = object.__new__
        # if a non-object.__new__ is used then whatever value/tuple was
        # assigned to the enum member name will be passed to __new__ and to the
        # new enum member's __init__
        if __new__ is object.__new__:
            new_uses_args = False
        else:
            new_uses_args = True
        #
        return __new__, save_new, new_uses_args


    # In order to support Python 2 and 3 with a single
    # codebase we have to create the Enum methods separately
    # and then use the `type(name, bases, dict)` method to
    # create the class.

EnumMeta = EnumType

enum_dict = _Addendum(
        dict=EnumType.__prepare__('Enum', (object, )),
        doc="Generic enumeration.\n\n    Derive from this class to define new enumerations.\n\n",
        ns=globals(),
        )

@enum_dict
def __init__(self, *args, **kwds):
    # auto-init method
    _auto_init_ = self._auto_init_
    if _auto_init_ is None:
        return
    if 'value' in _auto_init_:
        # remove 'value' from _auto_init_ as it has already been handled
        _auto_init_ = _auto_init_[1:]
    if _auto_init_:
        if len(_auto_init_) < len(args):
            raise TypeError('%d arguments expected (%s), %d received (%s)'
                    % (len(_auto_init_), _auto_init_, len(args), args))
        for name, arg in zip(_auto_init_, args):
            setattr(self, name, arg)
        if len(args) < len(_auto_init_):
            remaining_args = _auto_init_[len(args):]
            for name in remaining_args:
                value = kwds.pop(name, undefined)
                if value is undefined:
                    raise TypeError('missing value for: %r' % (name, ))
                setattr(self, name, value)
            if kwds:
                # too many keyword arguments
                raise TypeError('invalid keyword(s): %s' % ', '.join(kwds.keys()))

@enum_dict
def __new__(cls, value):
    # all enum instances are actually created during class construction
    # without calling this method; this method is called by the metaclass'
    # __call__ (i.e. Color(3) ), and by pickle
    if NoAlias in cls._settings_:
        raise TypeError('NoAlias enumerations cannot be looked up by value')
    if type(value) is cls:
        # For lookups like Color(Color.red)
        # value = value.value
        return value
    # by-value search for a matching enum member
    # see if it's in the reverse mapping (for hashable values)
    try:
        if value in cls._value2member_map_:
            return cls._value2member_map_[value]
    except TypeError:
        # not there, now do long search -- O(n) behavior
        for name, member in cls._value2member_seq_:
            if name == value:
                return member
    # still not found -- try _missing_ hook
    try:
        exc = None
        result = cls._missing_value_(value)
    except Exception as e:
        exc = e
        result = None
    if isinstance(result, cls) or getattr(cls, '_boundary_', None) is EJECT:
        return result
    else:
        if value is no_arg:
            ve_exc = ValueError('%s() should be called with a value' % (cls.__name__, ))
        else:
            ve_exc = ValueError("%r is not a valid %s" % (value, cls.__name__))
        if result is None and exc is None:
            raise ve_exc
        elif exc is None:
            exc = TypeError(
                    'error in %s._missing_: returned %r instead of None or a valid member'
                    % (cls.__name__, result)
                    )
        if not isinstance(exc, ValueError):
            exc.__cause__ = ve_exc
        raise exc

@enum_dict
@classmethod
def __init_subclass__(cls, **kwds):
    if pyver < PY3_6:
        # end of the line
        if kwds:
            raise TypeError('unconsumed keyword arguments: %r' % (kwds, ))
    else:
        super(Enum, cls).__init_subclass__(**kwds)

@enum_dict
@staticmethod
def _generate_next_value_(name, start, count, last_values, *args, **kwds):
    for last_value in reversed(last_values):
        try:
            new_value = last_value + 1
            break
        except TypeError:
            pass
    else:
        new_value = start
    if args:
        return (new_value, ) + args
    else:
        return new_value

@enum_dict
@classmethod
def _missing_(cls, value):
    "deprecated, use _missing_value_ instead"
    return None

@enum_dict
@classmethod
def _missing_value_(cls, value):
    "used for failed value access"
    return cls._missing_(value)

@enum_dict
@classmethod
def _missing_name_(cls, name):
    "used for failed item access"
    return None

@enum_dict
def __repr__(self):
    v_repr = self.__class__._value_repr_ or self._value_.__class__.__repr__
    return "<%s.%s: %s>" % (self.__class__.__name__, self._name_, v_repr(self._value_))

@enum_dict
def __str__(self):
    return "%s.%s" % (self.__class__.__name__, self._name_)

if PY3:
    @enum_dict
    def __dir__(self):
        """
        Returns all members and all public methods
        """
        if self.__class__._member_type_ is object:
            interesting = set(['__class__', '__doc__', '__eq__', '__hash__', '__module__', 'name', 'value'])
        else:
            interesting = set(object.__dir__(self))
        for name in getattr(self, '__dict__', []):
            if name[0] != '_':
                interesting.add(name)
        for cls in self.__class__.mro():
            for name, obj in cls.__dict__.items():
                if name[0] == '_':
                    continue
                if isinstance(obj, property):
                    # that's an enum.property
                    if obj.fget is not None or name not in self._member_map_:
                        interesting.add(name)
                    else:
                        # in case it was added by `dir(self)`
                        interesting.discard(name)
                else:
                    interesting.add(name)
        return sorted(interesting)

@enum_dict
def __format__(self, format_spec):
    # mixed-in Enums should use the mixed-in type's __format__, otherwise
    # we can get strange results with the Enum name showing up instead of
    # the value

    # pure Enum branch / overridden __str__ branch
    overridden_str = self.__class__.__str__ != Enum.__str__
    if self._member_type_ is object or overridden_str:
        cls = str
        val = str(self)
    # mix-in branch
    else:
        cls = self._member_type_
        val = self.value
    return cls.__format__(val, format_spec)

@enum_dict
def __hash__(self):
    return hash(self._name_)

@enum_dict
def __reduce_ex__(self, proto):
    return self.__class__, (self._value_, )


####################################
# Python's less than 2.6 use __cmp__

if pyver < PY2_6:

    @enum_dict
    def __cmp__(self, other):
        if type(other) is self.__class__:
            if self is other:
                return 0
            return -1
        return NotImplemented
        raise TypeError("unorderable types: %s() and %s()" % (self.__class__.__name__, other.__class__.__name__))

else:

    @enum_dict
    def __le__(self, other):
        raise TypeError("unorderable types: %s() <= %s()" % (self.__class__.__name__, other.__class__.__name__))

    @enum_dict
    def __lt__(self, other):
        raise TypeError("unorderable types: %s() < %s()" % (self.__class__.__name__, other.__class__.__name__))

    @enum_dict
    def __ge__(self, other):
        raise TypeError("unorderable types: %s() >= %s()" % (self.__class__.__name__, other.__class__.__name__))

    @enum_dict
    def __gt__(self, other):
        raise TypeError("unorderable types: %s() > %s()" % (self.__class__.__name__, other.__class__.__name__))


@enum_dict
def __eq__(self, other):
    if type(other) is self.__class__:
        return self is other
    return NotImplemented

@enum_dict
def __ne__(self, other):
    if type(other) is self.__class__:
        return self is not other
    return NotImplemented

@enum_dict
def __hash__(self):
    return hash(self._name_)

@enum_dict
def __reduce_ex__(self, proto):
    return self.__class__, (self._value_, )


# enum.property is used to provide access to the `name`, `value', etc.,
# properties of enum members while keeping some measure of protection
# from modification, while still allowing for an enumeration to have
# members named `name`, `value`, etc..  This works because enumeration
# members are not set directly on the enum class -- enum.property will
# look them up in _member_map_.

@enum_dict
@property
def name(self):
    return self._name_

@enum_dict
@property
def value(self):
    return self._value_

@enum_dict
@property
def values(self):
    return self._values_

def _reduce_ex_by_name(self, proto):
    return self.name

Enum = EnumType('Enum', (object, ), enum_dict.resolve())
del enum_dict

    # Enum has now been created

class ReprEnum(Enum):
    """
    Only changes the repr(), leaving str() and format() to the mixed-in type.
    """


class IntEnum(int, ReprEnum):
    """
    Enum where members are also (and must be) ints
    """


class StrEnum(str, ReprEnum):
    """
    Enum where members are also (and must already be) strings

    default value is member name, lower-cased
    """

    def __new__(cls, *values, **kwds):
        if kwds:
            raise TypeError('%r: keyword arguments not supported' % (cls.__name__))
        if values:
            if not isinstance(values[0], str):
                raise TypeError('%s: values must be str [%r is a %r]' % (cls.__name__, values[0], type(values[0])))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    __str__ = str.__str__

    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()


class LowerStrEnum(StrEnum):
    """
    Enum where members are also (and must already be) lower-case strings

    default value is member name, lower-cased
    """

    def __new__(cls, value, *args, **kwds):
        obj = StrEnum.__new_member__(cls, value, *args, **kwds)
        if value != value.lower():
            raise ValueError('%r is not lower-case' % value)
        return obj


class UpperStrEnum(StrEnum):
    """
    Enum where members are also (and must already be) upper-case strings

    default value is member name, upper-cased
    """

    def __new__(cls, value, *args, **kwds):
        obj = StrEnum.__new_member__(cls, value, *args, **kwds)
        if value != value.upper():
            raise ValueError('%r is not upper-case' % value)
        return obj

    def _generate_next_value_(name, start, count, last_values, *args, **kwds):
        return name.upper()


if PY3:
    class AutoEnum(Enum):
        """
        automatically use _generate_next_value_ when values are missing (Python 3 only)
        """
        _settings_ = MagicValue


class AutoNumberEnum(Enum):
    """
    Automatically assign increasing values to members.

    Py3: numbers match creation order
    Py2: numbers are assigned alphabetically by member name
         (unless `_order_` is specified)
    """

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        if cls._member_type_ is int:
            obj = int.__new__(cls, value)
        elif cls._member_type_ is long:
            obj = long.__new__(cls, value)
        else:
            obj = object.__new__(cls)
        obj._value_ = value
        return obj


class AddValueEnum(Enum):
    _settings_ = AddValue


class MultiValueEnum(Enum):
    """
    Multiple values can map to each member.
    """
    _settings_ = MultiValue


class NoAliasEnum(Enum):
    """
    Duplicate value members are distinct, but cannot be looked up by value.
    """
    _settings_ = NoAlias


class OrderedEnum(Enum):
    """
    Add ordering based on values of Enum members.
    """

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self._value_ >= other._value_
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self._value_ > other._value_
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self._value_ <= other._value_
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self._value_ < other._value_
        return NotImplemented


if sqlite3:
    class SqliteEnum(Enum):
        def __conform__(self, protocol):
            if protocol is sqlite3.PrepareProtocol:
                return self.name


class UniqueEnum(Enum):
    """
    Ensure no duplicate values exist.
    """
    _settings_ = Unique


def convert(enum, name, module, filter, source=None):
    """
    Create a new Enum subclass that replaces a collection of global constants

    enum: Enum, IntEnum, ...
    name: name of new Enum
    module: name of module (__name__ in global context)
    filter: function that returns True if name should be converted to Enum member
    source: namespace to check (defaults to 'module')
    """
    # convert all constants from source (or module) that pass filter() to
    # a new Enum called name, and export the enum and its members back to
    # module;
    # also, replace the __reduce_ex__ method so unpickling works in
    # previous Python versions
    module_globals = vars(_sys.modules[module])
    if source:
        source = vars(source)
    else:
        source = module_globals
    members = dict((name, value) for name, value in source.items() if filter(name))
    enum = enum(name, members, module=module)
    enum.__reduce_ex__ = _reduce_ex_by_name
    module_globals.update(enum.__members__)
    module_globals[name] = enum

def extend_enum(enumeration, name, *args, **kwds):
    """
    Add a new member to an existing Enum.
    """
    # there are four possibilities:
    # - extending an aenum Enum or 3.11+ enum Enum
    # - extending an aenum Flag or 3.11+ enum Flag
    # - extending a pre-3.11 stdlib Enum Flag
    # - extending a 3.11+ stdlib Flag
    #
    # fail early if name is already in the enumeration
    if (
            name in enumeration.__dict__
            or name in enumeration._member_map_
            or name in [t[1] for t in getattr(enumeration, '_value2member_seq_', ())]
        ):
        raise TypeError('%r already in use as %r' % (name, enumeration.__dict__.get(name, enumeration[name])))
    # and check for other instances in parent classes
    descriptor = None
    for base in enumeration.__mro__[1:]:
        descriptor = base.__dict__.get(name)
        if descriptor is not None:
            if isinstance(descriptor, (property, DynamicClassAttribute)):
                break
            else:
                raise TypeError('%r already in use in superclass %r' % (name, base.__name__))
    try:
        _member_map_ = enumeration._member_map_
        _member_names_ = enumeration._member_names_
        _member_type_ = enumeration._member_type_
        _value2member_map_ = enumeration._value2member_map_
        base_attributes = set([a for b in enumeration.mro() for a in b.__dict__])
    except AttributeError:
        raise TypeError('%r is not a supported Enum' % (enumeration, ))
    try:
        _value2member_seq_ = enumeration._value2member_seq_
        _multi_value_ = MultiValue in enumeration._settings_
        _no_alias_ = NoAlias in enumeration._settings_
        _unique_ = Unique in enumeration._settings_
        _auto_init_ = enumeration._auto_init_ or []
    except AttributeError:
        # standard Enum
        _value2member_seq_ = []
        _multi_value_ = False
        _no_alias_ = False
        _unique_ = False
        _auto_init_ = []
    if _multi_value_ and not args:
        # must specify values for multivalue enums
        raise ValueError('no values specified for MultiValue enum %r' % enumeration.__name__)
    mt_new = _member_type_.__new__
    _new = getattr(enumeration, '__new_member__', mt_new)
    if not args:
        last_values = [m.value for m in enumeration]
        count = len(enumeration)
        start = getattr(enumeration, '_start_', None)
        if start is None:
            start = last_values and (last_values[-1] + 1) or 1
        _gnv = getattr(enumeration, '_generate_next_value_', None)
        if _gnv is not None:
            args = ( _gnv(name, start, count, last_values), )
        else:
            # must be a 3.4 or 3.5 Enum
            args = (start, )
    if _new is object.__new__:
        new_uses_args = False
    else:
        new_uses_args = True
    if len(args) == 1:
        [value] = args
    else:
        value = args
    more_values = ()
    kwds = {}
    if isinstance(value, enum):
        args = value.args
        kwds = value.kwds
    if not isinstance(value, tuple):
        args = (value, )
    else:
        args = value
    # tease value out of auto-init if specified
    if 'value' in _auto_init_:
        if 'value' in kwds:
            value = kwds.pop('value')
        else:
            value, args = args[0], args[1:]
    elif _multi_value_:
        value, more_values, args = args[0], args[1:], ()
        if new_uses_args:
            args = (value, )
    if _member_type_ is tuple:
        args = (args, )
    if not new_uses_args:
        new_member = _new(enumeration)
        if not hasattr(new_member, '_value_'):
            new_member._value_ = value
    else:
        new_member = _new(enumeration, *args, **kwds)
        if not hasattr(new_member, '_value_'):
            new_member._value_ = _member_type_(*args)
    value = new_member._value_
    if _multi_value_:
        if 'value' in _auto_init_:
            args = more_values
        else:
        # put all the values back into args for the init call
            args = (value, ) + more_values
    new_member._name_ = name
    new_member.__objclass__ = enumeration.__class__
    new_member.__init__(*args)
    new_member._values_ = (value, ) + more_values
    # do final checks before modifying enum structures:
    # - is new member a flag?
    #   - does the new member fit in the enum's declared _boundary_?
    # - is new member an alias?
    #
    _all_bits_ = _flag_mask_ = None
    if hasattr(enumeration, '_all_bits_'):
        _all_bits_ = enumeration._all_bits_ | value
        _flag_mask_ = enumeration._flag_mask_ | value
        if enumeration._boundary_ != 'keep':
            missed = list(_iter_bits_lsb(_flag_mask_ & ~_all_bits_))
            if missed:
                raise TypeError(
                        'invalid Flag %r -- missing values: %s'
                        % (cls, ', '.join((str(i) for i in missed)))
                        )
    # If another member with the same value was already defined, the
    # new member becomes an alias to the existing one.
    if _no_alias_:
        # unless NoAlias was specified
        return _finalize_extend_enum(enumeration, new_member, bits=_all_bits_, mask=_flag_mask_)
    else:
        # handle "normal" aliases
        new_values = new_member._values_
        for canonical_member in _member_map_.values():
            canonical_values_ = getattr(canonical_member, '_values_', [canonical_member._value_])
            for canonical_value in canonical_values_:
                for new_value in new_values:
                    if canonical_value == new_value:
                        # name is an alias
                        if _unique_ or _multi_value_:
                            # aliases not allowed in Unique and MultiValue enums
                            raise ValueError('%r is a duplicate of %r' % (new_member, canonical_member))
                        else:
                            # aliased name can be added, remaining checks irrelevant
                            # aliases don't appear in member names (only in __members__ and _member_map_).
                            return _finalize_extend_enum(enumeration, canonical_member, name=name, bits=_all_bits_, mask=_flag_mask_, is_alias=True)
        # not a standard alias, but maybe a flag alias
        if pyver < PY3_6:
            flag_bases = Flag,
        else:
            flag_bases = Flag, StdlibFlag
        if issubclass(enumeration, flag_bases) and hasattr(enumeration, '_all_bits_'):
            # handle the new flag type
            if _is_single_bit(value):
                # a new member!  (an aliase would have been discovered in the previous loop)
                return _finalize_extend_enum(enumeration, new_member, bits=_all_bits_, mask=_flag_mask_)
            else:
                # might be an 3.11 Flag alias
                if value & enumeration._flag_mask_ == value and _value2member_map_.get(value) is not None:
                    # yup, it's an alias to existing members... and its an alias of an alias
                    canonical = _value2member_map_.get(value)
                    return _finalize_extend_enum(enumeration, canonical, name=name, bits=_all_bits_, mask=_flag_mask_, is_alias=True)
                else:
                    return _finalize_extend_enum(enumeration, new_member, bits=_all_bits_, mask=_flag_mask_, is_alias=True)
        else:
            # if we get here, we have a brand new member
            return _finalize_extend_enum(enumeration, new_member)

def _finalize_extend_enum(enumeration, new_member, name=None, bits=None, mask=None, is_alias=False):
    name = name or new_member.name
    descriptor = None
    for base in enumeration.__mro__[1:]:
        descriptor = base.__dict__.get(name)
        if descriptor is not None:
            if isinstance(descriptor, (property, DynamicClassAttribute)):
                break
            else:
                raise TypeError('%r already in use in superclass %r' % (name, base.__name__))
    if not descriptor:
        # get redirect in place before adding to _member_map_
        redirect = property()
        redirect.__set_name__(enumeration, name)
        setattr(enumeration, name, redirect)
    if not is_alias:
        enumeration._member_names_.append(name)
    enumeration._member_map_[name] = new_member
    for v in getattr(new_member, '_values_', [new_member._value_]):
        try:
            enumeration._value2member_map_[v] = new_member
        except TypeError:
            enumeration._value2member_seq_ += ((v, new_member), )
    if bits:
        enumeration._all_bits_ = bits
        enumeration._flag_mask_ = mask
    return new_member

def unique(enumeration):
    """
    Class decorator that ensures only unique members exist in an enumeration.
    """
    duplicates = []
    for name, member in enumeration.__members__.items():
        if name != member.name:
            duplicates.append((name, member.name))
    if duplicates:
        duplicate_names = ', '.join(
                ["%s -> %s" % (alias, name) for (alias, name) in duplicates]
                )
        raise ValueError('duplicate names found in %r: %s' %
                (enumeration, duplicate_names)
                )
    return enumeration

# Flag

@export(globals())
class FlagBoundary(StrEnum):
    """
    control how out of range values are handled
    "strict" -> error is raised  [default]
    "conform" -> extra bits are discarded
    "eject" -> lose flag status (becomes a normal integer)
    """
    STRICT = auto()
    CONFORM = auto()
    EJECT = auto()
    KEEP = auto()
assert FlagBoundary.STRICT == 'strict', (FlagBoundary.STRICT, FlagBoundary.CONFORM)

class Flag(Enum):
    """
    Generic flag enumeration.

    Derive from this class to define new flag enumerations.
    """

    _boundary_ = STRICT
    _numeric_repr_ = repr


    def _generate_next_value_(name, start, count, last_values, *args, **kwds):
        """
        Generate the next value when not given.

        name: the name of the member
        start: the initital start value or None
        count: the number of existing members
        last_value: the last value assigned or None
        """
        if not count:
            if args:
                return ((1, start)[start is not None], ) + args
            else:
                return (1, start)[start is not None]
        else:
            last_value = max(last_values)
            try:
                high_bit = _high_bit(last_value)
                result = 2 ** (high_bit+1)
                if args:
                    return (result,)  + args
                else:
                    return result
            except Exception:
                pass
            raise TypeError('invalid Flag value: %r' % last_value)

    @classmethod
    def _iter_member_by_value_(cls, value):
        """
        Extract all members from the value in definition (i.e. increasing value) order.
        """
        for val in _iter_bits_lsb(value & cls._flag_mask_):
            yield cls._value2member_map_.get(val)

    _iter_member_ = _iter_member_by_value_

    @classmethod
    def _iter_member_by_def_(cls, value):
        """
        Extract all members from the value in definition order.
        """
        members = list(cls._iter_member_by_value_(value))
        members.sort(key=lambda m: m._sort_order_)
        for member in members:
            yield member

    @classmethod
    def _missing_(cls, value):
        """
        return a member matching the given value, or None
        """
        return cls._create_pseudo_member_(value)

    @classmethod
    def _create_pseudo_member_(cls, *values):
        """
        Create a composite member.
        """
        value = values[0]
        if not isinstance(value, baseinteger):
            raise ValueError(
                    "%r is not a valid %s" % (value, getattr(cls, '__qualname__', cls.__name__))
                    )
        # check boundaries
        # - value must be in range (e.g. -16 <-> +15, i.e. ~15 <-> 15)
        # - value must not include any skipped flags (e.g. if bit 2 is not
        #   defined, then 0d10 is invalid)
        neg_value = None
        if (
                not ~cls._all_bits_ <= value <= cls._all_bits_
                or value & (cls._all_bits_ ^ cls._flag_mask_)
            ):
            if cls._boundary_ is STRICT:
                max_bits = max(value.bit_length(), cls._flag_mask_.bit_length())
                raise ValueError(
                        "%s: invalid value: %r\n    given %s\n  allowed %s"
                        % (cls.__name__, value, bin(value, max_bits), bin(cls._flag_mask_, max_bits))
                        )
            elif cls._boundary_ is CONFORM:
                value = value & cls._flag_mask_
            elif cls._boundary_ is EJECT:
                return value
            elif cls._boundary_ is KEEP:
                if value < 0:
                    value = (
                            max(cls._all_bits_+1, 2**(value.bit_length()))
                            + value
                            )
            else:
                raise ValueError(
                        'unknown flag boundary: %r' % (cls._boundary_, )
                        )
        if value < 0:
            neg_value = value
            value = cls._all_bits_ + 1 + value
        # get members and unknown
        unknown = value & ~cls._flag_mask_
        members = list(cls._iter_member_(value))
        if unknown and cls._boundary_ is not KEEP:
            raise ValueError(
                    '%s(%r) -->  unknown values %r [%s]'
                    % (cls.__name__, value, unknown, bin(unknown))
                    )
        # let class adjust values
        values = cls._create_pseudo_member_values_(members, *values)
        __new__ = getattr(cls, '__new_member__', None)
        if cls._member_type_ is object and not __new__:
            # construct a singleton enum pseudo-member
            pseudo_member = object.__new__(cls)
        else:
            pseudo_member = (__new__ or cls._member_type_.__new__)(cls, *values)
        if not hasattr(pseudo_member, 'value'):
            pseudo_member._value_ = value
        if members:
            pseudo_member._name_ = '|'.join([m._name_ for m in members])
            if unknown:
                pseudo_member._name_ += '|%s' % cls._numeric_repr_(unknown)
        else:
            pseudo_member._name_ = None
        # use setdefault in case another thread already created a composite
        # with this value, but only if all members are known
        # note: zero is a special case -- add it
        if not unknown:
            pseudo_member = cls._value2member_map_.setdefault(value, pseudo_member)
            if neg_value is not None:
                cls._value2member_map_[neg_value] = pseudo_member
        return pseudo_member


    @classmethod
    def _create_pseudo_member_values_(cls, members, *values):
        """
        Return values to be fed to __new__ to create new member.
        """
        if cls._member_type_ in (baseinteger + (object, )):
            return values
        elif len(values) < 2:
            return values + (cls._member_type_(), )
        else:
            return values

    def __contains__(self, other):
        """
        Returns True if self has at least the same flags set as other.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                "unsupported operand type(s) for 'in': '%s' and '%s'" % (
                    type(other).__name__, self.__class__.__name__))
        if other._value_ == 0 or self._value_ == 0:
            return False
        return other._value_ & self._value_ == other._value_

    def __iter__(self):
        """
        Returns flags in definition order.
        """
        for member in self._iter_member_(self._value_):
            yield member

    def __len__(self):
        return _bit_count(self._value_)

    def __repr__(self):
        cls = self.__class__
        if self._name_ is None:
            # only zero is unnamed by default
            return '<%s: %r>' % (cls.__name__, self._value_)
        else:
            return '<%s.%s: %r>' % (cls.__name__, self._name_, self._value_)

    def __str__(self):
        cls = self.__class__
        if self._name_ is None:
            return '%s(%s)' % (cls.__name__, self._value_)
        else:
            return '%s.%s' % (cls.__name__, self._name_)

    if PY2:
        def __nonzero__(self):
            return bool(self._value_)
    else:
        def __bool__(self):
            return bool(self._value_)

    def __or__(self, other):
        if isinstance(other, self.__class__):
            other_value = other._value_
        elif self._member_type_ is not object and isinstance(other, self._member_type_):
            other_value = other
        else:
            return NotImplemented
        return self.__class__(self._value_ | other_value)

    def __and__(self, other):
        if isinstance(other, self.__class__):
            other_value = other._value_
        elif self._member_type_ is not object and isinstance(other, self._member_type_):
            other_value = other
        else:
            return NotImplemented
        return self.__class__(self._value_ & other_value)

    def __xor__(self, other):
        if isinstance(other, self.__class__):
            other_value = other._value_
        elif self._member_type_ is not object and isinstance(other, self._member_type_):
            other_value = other
        else:
            return NotImplemented
        return self.__class__(self._value_ ^ other_value)

    def __invert__(self):
        if self._inverted_ is None:
            if self._boundary_ is KEEP:
                # use all bits
                self._inverted_ = self.__class__(~self._value_)
            else:
                # calculate flags not in this member
                self._inverted_ = self.__class__(self._flag_mask_ ^ self._value_)
            self._inverted_._inverted_ = self
        return self._inverted_

    __ror__ = __or__
    __rand__ = __and__
    __rxor__ = __xor__



class IntFlag(int, ReprEnum, Flag):
    """Support for integer-based Flags"""

    _boundary_ = EJECT


def _high_bit(value):
    """returns index of highest bit, or -1 if value is zero or negative"""
    return value.bit_length() - 1

def global_enum_repr(self):
    """
    use module.enum_name instead of class.enum_name

    the module is the last module in case of a multi-module name
    """
    module = self.__class__.__module__.split('.')[-1]
    return '%s.%s' % (module, self._name_)

def global_flag_repr(self):
    """
    use module.flag_name instead of class.flag_name

    the module is the last module in case of a multi-module name
    """
    module = self.__class__.__module__.split('.')[-1]
    cls_name = self.__class__.__name__
    if self._name_ is None:
        return "%s.%s(%r)" % (module, cls_name, self._value_)
    if _is_single_bit(self):
        return '%s.%s' % (module, self._name_)
    if self._boundary_ is not FlagBoundary.KEEP:
        return '|'.join(['%s.%s' % (module, name) for name in self.name.split('|')])
    else:
        name = []
        for n in self._name_.split('|'):
            if n[0].isdigit():
                name.append(n)
            else:
                name.append('%s.%s' % (module, n))
        return '|'.join(name)

def global_str(self):
    """
    use enum_name instead of class.enum_name
    """
    if self._name_ is None:
        return "%s(%r)" % (cls_name, self._value_)
    else:
        return self._name_

def global_enum(cls, update_str=False):
    """
    decorator that makes the repr() of an enum member reference its module
    instead of its class; also exports all members to the enum's module's
    global namespace
    """
    if issubclass(cls, Flag):
        cls.__repr__ = global_flag_repr
    else:
        cls.__repr__ = global_enum_repr
    if not issubclass(cls, ReprEnum) or update_str:
        cls.__str__ = global_str
    _sys.modules[cls.__module__].__dict__.update(cls.__members__)
    return cls


class module(object):

    def __init__(self, cls, *args):
        self.__name__ = cls.__name__
        self._parent_module = cls.__module__
        self.__all__ = []
        all_objects = cls.__dict__
        if not args:
            args = [k for k, v in all_objects.items() if isinstance(v, (NamedConstant, Enum))]
        for name in args:
            self.__dict__[name] = all_objects[name]
            self.__all__.append(name)

    def register(self):
        _sys.modules["%s.%s" % (self._parent_module, self.__name__)] = self

if StdlibEnumMeta:

    from _weakrefset import WeakSet

    def __subclasscheck__(cls, subclass):
        """
        Override for issubclass(subclass, cls).
        """
        if not isinstance(subclass, type):
            raise TypeError('issubclass() arg 1 must be a class (got %r)' % (subclass, ))
        # Check cache
        try:
            cls.__dict__['_subclass_cache_']
        except KeyError:
            cls._subclass_cache_ = WeakSet()
            cls._subclass_negative_cache_ = WeakSet()
        except RecursionError:
            import sys
            exc, cls, tb = sys.exc_info()
            exc = RecursionError('possible causes for endless recursion:\n    - __getattribute__ is not ignoring __dunder__ attibutes\n    - __instancecheck__ and/or __subclasscheck_ are (mutually) recursive\n    see `aenum.remove_stdlib_integration` for temporary work-around')
            raise_from_none(exc)
        if subclass in cls._subclass_cache_:
            return True
        # Check negative cache
        elif subclass in cls._subclass_negative_cache_:
            return False
        if cls is subclass:
            cls._subclass_cache_.add(subclass)
            return True
        # Check if it's a direct subclass
        if cls in getattr(subclass, '__mro__', ()):
            cls._subclass_cache_.add(subclass)
            return True
        # Check if it's an aenum.Enum|IntEnum|IntFlag|Flag subclass
        if cls is StdlibIntFlag and issubclass(subclass, IntFlag):
            cls._subclass_cache_.add(subclass)
            return True
        elif cls is StdlibFlag and issubclass(subclass, Flag):
            cls._subclass_cache_.add(subclass)
            return True
        elif cls is StdlibIntEnum and issubclass(subclass, IntEnum):
            cls._subclass_cache_.add(subclass)
            return True
        if cls is StdlibEnum and issubclass(subclass, Enum):
            cls._subclass_cache_.add(subclass)
            return True
        # No dice; update negative cache
        cls._subclass_negative_cache_.add(subclass)
        return False

    def __instancecheck__(cls, instance):
        subclass = instance.__class__
        try:
            return cls.__subclasscheck__(subclass)
        except RecursionError:
            import sys
            exc, cls, tb = sys.exc_info()
            exc = RecursionError('possible causes for endless recursion:\n    - __getattribute__ is not ignoring __dunder__ attibutes\n    - __instancecheck__ and/or __subclasscheck_ are (mutually) recursive\n    see `aenum.remove_stdlib_integration` for temporary work-around')
            raise_from_none(exc)

    StdlibEnumMeta.__subclasscheck__ = __subclasscheck__
    StdlibEnumMeta.__instancecheck__ = __instancecheck__

def add_stdlib_integration():
    if StdlibEnum:
        StdlibEnumMeta.__subclasscheck__ = __subclasscheck__
        StdlibEnumMeta.__instancecheck__ = __instancecheck__

def remove_stdlib_integration():
    """
    Remove the __instancecheck__ and __subclasscheck__ overrides from the stdlib Enum.

    Those overrides are in place so that code detecting stdlib enums will also detect
    aenum enums.  If a buggy __getattribute__, __instancecheck__, or __subclasscheck__
    is defined on a custom EnumMeta then RecursionErrors can result; using this
    function after importing aenum will solve that problem, but the better solution is
    to fix the buggy method.
    """
    if StdlibEnum:
        del StdlibEnumMeta.__instancecheck__
        del StdlibEnumMeta.__subclasscheck__

