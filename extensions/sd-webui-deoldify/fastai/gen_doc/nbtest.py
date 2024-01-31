"`gen_doc.nbtest` shows pytest documentation for module functions"

import inspect, os, re
from os.path import abspath, dirname, join
from collections import namedtuple

from fastai.gen_doc import nbdoc
from ..imports.core import *
from .core import ifnone
from .doctest import get_parent_func, relative_test_path, get_func_fq_name, DB_NAME

from nbconvert import HTMLExporter
from IPython.core import page
from IPython.core.display import display, Markdown, HTML

__all__ = ['show_test', 'doctest', 'find_related_tests', 'lookup_db', 'find_test_matches', 'find_test_files', 'fuzzy_test_match', 'get_pytest_html']

TestFunctionMatch = namedtuple('TestFunctionMatch', ['line_number', 'line'])

def show_test(elt)->str:
    "Show associated tests for a fastai function/class"
    md = build_tests_markdown(elt)
    display(Markdown(md))

def doctest(elt):
    "Inline notebook popup for `show_test`"
    md = build_tests_markdown(elt)
    output = nbdoc.md2html(md)
    try:    page.page({'text/html': output})
    except: display(Markdown(md))

def build_tests_markdown(elt):
    fn_name = nbdoc.fn_name(elt)
    md = ''
    db_matches = [get_links(t) for t in lookup_db(elt)]
    md += tests2md(db_matches, '')
    try:
        related = [get_links(t) for t in find_related_tests(elt)]
        other_tests = [k for k in OrderedDict.fromkeys(related) if k not in db_matches]
        md += tests2md(other_tests, f'Some other tests where `{fn_name}` is used:')
    except OSError as e: pass

    if len(md.strip())==0:
        return (f'No tests found for `{fn_name}`.'
                ' To contribute a test please refer to [this guide](/dev/test.html)'
                ' and [this discussion](https://forums.fast.ai/t/improving-expanding-functional-tests/32929).')
    return (f'Tests found for `{fn_name}`: {md}'
            '\n\nTo run tests please refer to this [guide](/dev/test.html#quick-guide).')

def tests2md(tests, type_label:str):
    if not tests: return ''
    md = [f'\n\n{type_label}'] + [f'* `{cmd}` {link}' for link,cmd in sorted(tests, key=lambda k: k[1])]
    return '\n'.join(md)

def get_pytest_html(elt, anchor_id:str)->Tuple[str,str]:
    md = build_tests_markdown(elt)
    html = nbdoc.md2html(md).replace('\n','') # nbconverter fails to parse markdown if it has both html and '\n'
    anchor_id = anchor_id.replace('.', '-') + '-pytest'
    link, body = get_pytest_card(html, anchor_id)
    return link, body

def get_pytest_card(html, anchor_id):
    "creates a collapsible bootstrap card for `show_test`"
    link = f'<a class="source_link" data-toggle="collapse" data-target="#{anchor_id}" style="float:right; padding-right:10px">[test]</a>'
    body = (f'<div class="collapse" id="{anchor_id}"><div class="card card-body pytest_card">'
                f'<a type="button" data-toggle="collapse" data-target="#{anchor_id}" class="close" aria-label="Close"><span aria-hidden="true">&times;</span></a>'
                f'{html}'
            '</div></div>')
    return link, body

def lookup_db(elt)->List[Dict]:
    "Finds `this_test` entries from test_registry.json"
    db_file = Path(abspath(join(dirname( __file__ ), '..')))/DB_NAME
    if not db_file.exists():
        raise Exception(f'Could not find {db_file}. Please make sure it exists at "{db_file}" or run `make test`')
    with open(db_file, 'r') as f:
        db = json.load(f)
    key = get_func_fq_name(elt)
    return db.get(key, [])

def find_related_tests(elt)->Tuple[List[Dict],List[Dict]]:
    "Searches `fastai/tests` folder for any test functions related to `elt`"
    related_matches = []
    for test_file in find_test_files(elt):
        fuzzy_matches = find_test_matches(elt, test_file)
        related_matches.extend(fuzzy_matches)
    return related_matches

def get_tests_dir(elt)->Path:
    "Absolute path of `fastai/tests` directory"
    test_dir = Path(__file__).parent.parent.parent.resolve()/'tests'
    if not test_dir.exists(): raise OSError('Could not find test directory at this location:', test_dir)
    return test_dir

def get_file(elt)->str:
    if hasattr(elt, '__wrapped__'): elt = elt.__wrapped__
    if not nbdoc.is_fastai_class(elt): return None
    return inspect.getfile(elt)

def find_test_files(elt, exact_match:bool=False)->List[Path]:
    "Searches in `fastai/tests` directory for module tests"
    test_dir = get_tests_dir(elt)
    matches = [test_dir/o.name for o in os.scandir(test_dir) if _is_file_match(elt, o.name)]
    # if len(matches) != 1: raise Error('Could not find exact file match:', matches)
    return matches

def _is_file_match(elt, file_name:str, exact_match:bool=False)->bool:
    fp = get_file(elt)
    if fp is None: return False
    subdir = ifnone(_submodule_name(elt), '')
    exact_re = '' if exact_match else '\w*'
    return re.match(f'test_{subdir}\w*{Path(fp).stem}{exact_re}\.py', file_name)

def _submodule_name(elt)->str:
    "Returns submodule - utils, text, vision, imports, etc."
    if inspect.ismodule(elt): return None
    modules = elt.__module__.split('.')
    if len(modules) > 2:
        return modules[1]
    return None

def find_test_matches(elt, test_file:Path)->Tuple[List[Dict],List[Dict]]:
    "Find all functions in `test_file` related to `elt`"
    lines = get_lines(test_file)
    rel_path = relative_test_path(test_file)
    fn_name = get_qualname(elt) if not inspect.ismodule(elt) else ''
    return fuzzy_test_match(fn_name, lines, rel_path)

def get_qualname(elt):
    return elt.__qualname__ if hasattr(elt, '__qualname__') else fn_name(elt)

def separate_comp(qualname:str):
    if not isinstance(qualname, str): qualname = get_qualname(qualname)
    parts = qualname.split('.')
    parts[-1] = remove_underscore(parts[-1])
    if len(parts) == 1: return [], parts[0]
    return parts[:-1], parts[-1]

def remove_underscore(fn_name):
    if fn_name and fn_name[0] == '_': return fn_name[1:] # remove private method underscore prefix
    return fn_name

def fuzzy_test_match(fn_name:str, lines:List[Dict], rel_path:str)->List[TestFunctionMatch]:
    "Find any lines where `fn_name` is invoked and return the parent test function"
    fuzzy_line_matches = _fuzzy_line_match(fn_name, lines)
    fuzzy_matches = [get_parent_func(lno, lines, ignore_missing=True) for lno,_ in fuzzy_line_matches]
    fuzzy_matches = list(filter(None.__ne__, fuzzy_matches))
    return [map_test(rel_path, lno, l) for lno,l in fuzzy_matches]

def _fuzzy_line_match(fn_name:str, lines)->List[TestFunctionMatch]:
    "Find any lines where `fn_name` is called"
    result = []
    _,fn_name = separate_comp(fn_name)
    for idx,line in enumerate(lines):
        if re.match(f'.*[\s\.\(]{fn_name}[\.\(]', line):
            result.append((idx,line))
    return result

def get_lines(file:Path)->List[str]:
    with open(file, 'r') as f: return f.readlines()

def map_test(test_file, line, line_text):
    "Creates dictionary test format to match doctest api"
    test_name = re.match(f'\s*def (test_\w*)', line_text).groups(0)[0]
    return { 'file': test_file, 'line': line, 'test': test_name }

def get_links(metadata)->Tuple[str,str]:
    "Returns source code link and pytest command"
    return nbdoc.get_source_link(**metadata), pytest_command(**metadata)

def pytest_command(file:str, test:str, **kwargs)->str:
    "Returns CLI command to run specific test function"
    return f'pytest -sv {file}::{test}'
