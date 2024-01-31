import sys, re, json, pprint
from pathlib import Path
from collections import defaultdict
from inspect import currentframe, getframeinfo, ismodule

__all__ = ['this_tests']

DB_NAME = 'test_registry.json'

def _json_set_default(obj):
    if isinstance(obj, set): return list(obj)
    raise TypeError

class TestRegistry:
    "Tests register which API they validate using this class."
    registry = defaultdict(list)
    this_tests_check = None
    missing_this_tests = set()

    # logic for checking whether each test calls `this_tests`:
    # 1. `this_tests_check` is set to True during test's 'setup' stage if it wasn't skipped
    # 2. if the test is dynamically skipped `this_tests_check` is set to False
    # 3. `this_tests` sets this flag to False when it's successfully completes
    # 4. if during the 'teardown' stage `this_tests_check` is still True then we
    # know that this test needs `this_tests_check`

    @staticmethod
    def this_tests(*funcs):
        prev_frame = currentframe().f_back.f_back
        file_name, lineno, test_name, _, _ = getframeinfo(prev_frame)
        parent_func_lineno, _ = get_parent_func(lineno, get_lines(file_name))
        entry = {'file': relative_test_path(file_name), 'test': test_name , 'line': parent_func_lineno}
        for func in funcs:
            if func == 'na':
                # special case when we can't find a function to declare, e.g.
                # when attributes are tested
                continue
            try:
                func_fq = get_func_fq_name(func)
            except:
                raise Exception(f"'{func}' is not a function") from None
            if re.match(r'fastai\.', func_fq):
                if entry not in TestRegistry.registry[func_fq]:
                    TestRegistry.registry[func_fq].append(entry)
            else:
                raise Exception(f"'{func}' is not in the fastai API") from None
        TestRegistry.this_tests_check = False

    def this_tests_check_on():
        TestRegistry.this_tests_check = True

    def this_tests_check_off():
        TestRegistry.this_tests_check = False

    def this_tests_check_run(file_name, test_name):
        if TestRegistry.this_tests_check:
            TestRegistry.missing_this_tests.add(f"{file_name}::{test_name}")

    def registry_save():
        if TestRegistry.registry:
            path = Path(__file__).parent.parent.resolve()/DB_NAME
            if path.exists():
                #print("\n*** Merging with the existing test registry")
                with open(path, 'r') as f: old_registry = json.load(f)
                TestRegistry.registry = merge_registries(old_registry, TestRegistry.registry)
            #print(f"\n*** Saving test registry @ {path}")
            with open(path, 'w') as f:
                json.dump(obj=TestRegistry.registry, fp=f, indent=4, sort_keys=True, default=_json_set_default)

    def missing_this_tests_alert():
        if TestRegistry.missing_this_tests:
            tests = '\n  '.join(sorted(TestRegistry.missing_this_tests))
            print(f"""
*** Attention ***
Please include `this_tests` call in each of the following tests:
  {tests}
For details see: https://docs.fast.ai/dev/test.html#test-registry""")

# merge_registries helpers
# merge dict of lists of dict
def a2k(a): return '::'.join([a['file'], a['test']]), a['line']
def k2a(k, v): f,t = k.split('::'); return {"file": f, "line": v, "test": t}
# merge by key that is a combination of 2 values: test, file
def merge_lists(a, b):
    x = dict(map(a2k, [*a, *b]))            # pack + merge
    return [k2a(k, v) for k,v in x.items()] # unpack
def merge_registries(a, b):
    for i in b: a[i] = merge_lists(a[i], b[i]) if i in a else b[i]
    return a

def this_tests(*funcs): TestRegistry.this_tests(*funcs)

def str2func(name):
    "Converts 'fastai.foo.bar' into an function 'object' if such exists"
    if isinstance(name, str): subpaths = name.split('.')
    else:                     return None

    module = subpaths.pop(0)
    if module in sys.modules: obj = sys.modules[module]
    else:                     return None

    for subpath in subpaths:
        obj = getattr(obj, subpath, None)
        if obj == None: return None
    return obj

def get_func_fq_name(func):
    if ismodule(func): return func.__name__
    if isinstance(func, str): func = str2func(func)
    name = None
    if   hasattr(func, '__qualname__'): name = func.__qualname__
    elif hasattr(func, '__name__'):     name = func.__name__
    elif hasattr(func, '__wrapped__'):  return get_func_fq_name(func.__wrapped__)
    elif hasattr(func, '__class__'):    name = func.__class__.__name__
    else: raise Exception(f"'{func}' is not a func or class")
    return f'{func.__module__}.{name}'

def get_parent_func(lineno, lines, ignore_missing=False):
    "Find any lines where `elt` is called and return the parent test function"
    for idx,l in enumerate(reversed(lines[:lineno])):
        if re.match(f'\s*def test', l):  return (lineno - idx), l # 1 based index for github
        if re.match(f'\w+', l):  break # top level indent - out of function scope
    if ignore_missing: return None
    raise LookupError('Could not find parent function for line:', lineno, lines[:lineno])

def relative_test_path(test_file:Path)->str:
    "Path relative to the `fastai` parent directory"
    test_file = Path(test_file)
    testdir_idx = list(reversed(test_file.parts)).index('tests')
    return '/'.join(test_file.parts[-(testdir_idx+1):])

def get_lines(file):
    with open(file, 'r') as f: return f.readlines()
