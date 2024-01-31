"`gen_doc.nbdoc` generates notebook documentation from module functions and links to correct places"
import pkgutil, inspect, sys,os, importlib,json,enum,warnings,nbformat,re
from IPython.core.display import display, Markdown
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.sign import NotebookNotary
from pathlib import Path
from .core import *
from .nbdoc import *

__all__ = ['create_module_page', 'update_module_page', 'import_mod',
           'link_nb', 'update_notebooks', 'generate_missing_metadata', 'update_nb_metadata']

def get_empty_notebook():
    "Default notbook with the minimum metadata."
    #TODO: check python version and nbformat
    return {'metadata': {'kernelspec': {'display_name': 'Python 3',
                                        'language': 'python',
                                        'name': 'python3'},
                         'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3},
                         'file_extension': '.py',
                         'mimetype': 'text/x-python',
                         'name': 'python',
                         'nbconvert_exporter': 'python',
                         'pygments_lexer': 'ipython3',
                         'version': '3.6.6'}},
            'nbformat': 4,
            'nbformat_minor': 2}

def get_md_cell(source, metadata=None):
    "Markdown cell containing `source` with `metadata`."
    return {'cell_type': 'markdown',
            'metadata': {} if metadata is None else metadata,
            'source': source}

def get_empty_cell(ctype='markdown'):
    "Empty cell of type `ctype`."
    return {'cell_type': ctype, 'metadata': {}, 'source': []}

def get_code_cell(code, hidden=False):
    "Code cell containing `code` that may be `hidden`."
    return {'cell_type' : 'code',
            'execution_count': 0,
            'metadata' : {'hide_input': hidden, 'trusted':True},
            'source' : code,
            'outputs': []}

def get_doc_cell(func_name):
    "Code cell with the command to show the doc of `func_name`."
    code = f"show_doc({func_name})"
    return get_code_cell(code, True)

def get_global_vars(mod):
    "Return globally assigned variables."
    # https://stackoverflow.com/questions/8820276/docstring-for-variable/31764368#31764368
    import ast,re
    with open(mod.__file__, 'r') as f: fstr = f.read()
    flines = fstr.splitlines()
    d = {}
    for node in ast.walk(ast.parse(fstr)):
        if isinstance(node,ast.Assign) and hasattr(node.targets[0], 'id'):
            key,lineno = node.targets[0].id,node.targets[0].lineno
            codestr = flines[lineno]
            match = re.match(f"^({key})\s*=\s*.*", codestr)
            if match and match.group(1) != '__all__': # only top level assignment
                d[key] = f'`{codestr}` {get_source_link(mod, lineno)}'
    return d

def write_nb(nb, nb_path, mode='w'):
    with open(nb_path, mode) as f: f.write(nbformat.writes(nbformat.from_dict(nb), version=4))

class ExecuteShowDocPreprocessor(ExecutePreprocessor):
    "An ExecutePreprocessor that only executes show_doc cells"
    def preprocess_cell(self, cell, resources, index):
        if 'source' in cell and cell.cell_type == "code":
            if IMPORT_RE.search(cell['source']) or SHOW_DOC_RE.search(cell['source']):
                return super().preprocess_cell(cell, resources, index)
        return cell, resources

def execute_nb(fname, metadata=None, save=True, show_doc_only=False):
    "Execute notebook `fname` with `metadata` for preprocessing."
    # Any module used in the notebook that isn't inside must be in the same directory as this script
    with open(fname) as f: nb = nbformat.read(f, as_version=4)
    ep_class = ExecuteShowDocPreprocessor if show_doc_only else ExecutePreprocessor
    ep = ep_class(timeout=600, kernel_name='python3')
    metadata = metadata or {}
    ep.preprocess(nb, metadata)
    if save:
        with open(fname, 'wt') as f: nbformat.write(nb, f)
        NotebookNotary().sign(nb)

def _symbol_skeleton(name): return [get_doc_cell(name), get_md_cell(f"`{name}`")]

def create_module_page(mod, dest_path, force=False):
    "Create the documentation notebook for module `mod_name` in path `dest_path`"
    nb = get_empty_notebook()
    mod_name = mod.__name__
    strip_name = strip_fastai(mod_name)
    init_cell = [get_md_cell(f'## Title for {strip_name} (use plain english, not module name!)'), get_md_cell('Type an introduction of the package here.')]
    cells = [get_code_cell(f'from fastai.gen_doc.nbdoc import *\nfrom {mod_name} import * ', True)]

    gvar_map = get_global_vars(mod)
    if gvar_map: cells.append(get_md_cell('### Global Variable Definitions:'))
    for name in get_exports(mod):
        if name in gvar_map: cells.append(get_md_cell(gvar_map[name]))

    for ft_name in get_ft_names(mod, include_inner=True):
        if not hasattr(mod, ft_name):
            warnings.warn(f"Module {strip_name} doesn't have a function named {ft_name}.")
            continue
        cells += _symbol_skeleton(ft_name)
        elt = getattr(mod, ft_name)
    nb['cells'] = init_cell + cells + [get_md_cell(UNDOC_HEADER)]

    doc_path = get_doc_path(mod, dest_path)
    write_nb(nb, doc_path, 'w' if force else 'x')
    execute_nb(doc_path)
    return doc_path

_default_exclude = ['.ipynb_checkpoints', '__pycache__', '__init__.py', 'imports']

def get_module_names(path_dir, exclude=None):
    if exclude is None: exclude = _default_exclude
    "Search a given `path_dir` and return all the modules contained inside except those in `exclude`"
    files = sorted(path_dir.glob('*'), key=lambda x: (x.is_dir(), x.name), reverse=True) # directories first
    res = [f'{path_dir.name}']
    for f in files:
        if f.is_dir() and f.name in exclude: continue # exclude directories
        if any([f.name.endswith(ex) for ex in exclude]): continue # exclude extensions

        if f.suffix == '.py': res.append(f'{path_dir.name}.{f.stem}')
        elif f.is_dir(): res += [f'{path_dir.name}.{name}' for name in get_module_names(f)]
    return res

def read_nb(fname):
    "Read a notebook in `fname` and return its corresponding json"
    with open(fname,'r') as f: return nbformat.reads(f.read(), as_version=4)

SHOW_DOC_RE = re.compile(r"show_doc\(([\w\.]*)")
def read_nb_content(cells, mod_name):
    "Build a dictionary containing the position of the `cells`."
    doc_fns = {}
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            for match in SHOW_DOC_RE.findall(cell['source']):
                doc_fns[match] = i
    return doc_fns

def read_nb_types(cells):
    doc_fns = {}
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            match = re.match(r"^(?:<code>|`)?(\w*)\s*=\s*", cell['source'])
            if match is not None: doc_fns[match.group(1)] = i
    return doc_fns

def link_markdown_cells(cells, modules):
    "Create documentation links for all cells in markdown with backticks."
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            cell['source'] = link_docstring(modules, cell['source'])

def get_insert_idx(pos_dict, name):
    "Return the position to insert a given function doc in a notebook."
    keys,i = list(pos_dict.keys()),0
    while i < len(keys) and str.lower(keys[i]) < str.lower(name): i+=1
    if i == len(keys): return -1
    else:              return pos_dict[keys[i]]

def update_pos(pos_dict, start_key, nbr=2):
    "Update the `pos_dict` by moving all positions after `start_key` by `nbr`."
    for key,idx in pos_dict.items():
        if str.lower(key) >= str.lower(start_key): pos_dict[key] += nbr
    return pos_dict

def insert_cells(cells, pos_dict, ft_name, append=False):
    "Insert the function doc `cells` at their correct position and updates `pos_dict`."
    idx = get_insert_idx(pos_dict, ft_name)
    if append or idx == -1: cells += [get_doc_cell(ft_name), get_empty_cell()]
    else:
        cells.insert(idx, get_doc_cell(ft_name))
        cells.insert(idx+1, get_empty_cell())
        pos_dict = update_pos(pos_dict, ft_name, 2)
    return cells, pos_dict

def get_doc_path(mod, dest_path):
    strip_name = strip_fastai(mod.__name__)
    return os.path.join(dest_path,f'{strip_name}.ipynb')

def generate_missing_metadata(dest_file):
    fn = Path(dest_file)
    meta_fn = fn.parent/'jekyll_metadata.ipynb'
    if not fn.exists() or not meta_fn.exists(): return print('Could not find notebooks:', fn, meta_fn)
    metadata_nb = read_nb(meta_fn)

    if has_metadata_cell(metadata_nb['cells'], fn.name): return
    nb = read_nb(fn)
    jmd = nb['metadata'].get('jekyll', {})
    fmt_params = ''
    for k,v in jmd.items(): fmt_params += f',\n    {k}={stringify(v)}'
    metadata_cell = get_code_cell(f"update_nb_metadata('{Path(fn).name}'{fmt_params})", hidden=False)
    metadata_nb['cells'].append(metadata_cell)
    write_nb(metadata_nb, meta_fn)

def update_nb_metadata(nb_path=None, title=None, summary=None, keywords='fastai', overwrite=True, **kwargs):
    "Creates jekyll metadata for given notebook path."
    nb = read_nb(nb_path)
    data = {'title': title, 'summary': summary, 'keywords': keywords, **kwargs}
    data = {k:v for (k,v) in data.items() if v is not None} # remove none values
    if not data: return
    nb['metadata']['jekyll'] = data
    write_nb(nb, nb_path)
    NotebookNotary().sign(nb)

def has_metadata_cell(cells, fn):
    for c in cells:
        if re.search(f"update_nb_metadata\('{fn}'", c['source']): return c

def stringify(s): return f'\'{s}\'' if isinstance(s, str) else s

IMPORT_RE = re.compile(r"from (fastai[\.\w_]*)")
def get_imported_modules(cells, nb_module_name=''):
    "Finds all submodules of notebook - sorted by submodules > top level modules > manual imports. This gives notebook imports priority"
    module_names = get_top_level_modules()
    nb_imports = [match.group(1) for cell in cells for match in IMPORT_RE.finditer(cell['source']) if cell['cell_type'] == 'code']
    parts = nb_module_name.split('.')
    parent_modules = ['.'.join(parts[:(x+1)]) for x in range_of(parts)] # Imports parent modules - a.b.c = [a, a.b, a.b.c]
    all_modules = module_names + nb_imports + parent_modules
    mods = [import_mod(m, ignore_errors=True) for m in all_modules]
    return [m for m in mods if m is not None]

def get_top_level_modules(num_levels=1):
    mod_dir = Path(import_mod('fastai').__file__).parent
    filtered_n = filter(lambda x: x.count('.')<=num_levels, get_module_names(mod_dir))
    return sorted(filtered_n, key=lambda s: s.count('.'), reverse=True) # Submodules first (sorted by periods)

NEW_FT_HEADER = '## New Methods - Please document or move to the undocumented section'
UNDOC_HEADER = '## Undocumented Methods - Methods moved below this line will intentionally be hidden'
def parse_sections(cells):
    old_cells, undoc_cells, new_cells = [], [], []
    current_section = old_cells
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            if re.match(UNDOC_HEADER, cell['source']): current_section = undoc_cells
            if re.match(NEW_FT_HEADER, cell['source']): current_section = new_cells
        current_section.append(cell)
    undoc_cells = undoc_cells or [get_md_cell(UNDOC_HEADER)]
    new_cells = new_cells or [get_md_cell(NEW_FT_HEADER)]
    return old_cells, undoc_cells, new_cells

def remove_undoc_cells(cells):
    old, _, _ = parse_sections(cells)
    return old

# currently code vbox sub-cells mainly
def remove_code_cell_jupyter_widget_state_elem(cells):
    for c in cells:
        if c['cell_type'] == 'code':
            if 'outputs' in c:
                c['outputs'] = [l for l in c['outputs'] if not ('data' in l and 'application/vnd.jupyter.widget-view+json' in l.data)]
    return cells

def update_module_page(mod, dest_path='.'):
    "Update the documentation notebook of a given module."
    doc_path = get_doc_path(mod, dest_path)
    strip_name = strip_fastai(mod.__name__)
    nb = read_nb(doc_path)
    cells = nb['cells']

    link_markdown_cells(cells, get_imported_modules(cells, mod.__name__))

    type_dict = read_nb_types(cells)
    gvar_map = get_global_vars(mod)
    for name in get_exports(mod):
        if name not in gvar_map: continue
        code = gvar_map[name]
        if name in type_dict: cells[type_dict[name]] = get_md_cell(code)
        else: cells.append(get_md_cell(code))

    pos_dict = read_nb_content(cells, strip_name)
    ft_names = get_ft_names(mod, include_inner=True)
    new_fts = list(set(ft_names) - set(pos_dict.keys()))
    if new_fts: print(f'Found new fuctions for {mod}. Please document:\n{new_fts}')
    existing, undoc_cells, new_cells = parse_sections(cells)
    for ft_name in new_fts: new_cells.extend([get_doc_cell(ft_name), get_empty_cell()])
    if len(new_cells) > 1: nb['cells'] = existing + undoc_cells + new_cells

    write_nb(nb, doc_path)
    return doc_path

def link_nb(nb_path):
    nb = read_nb(nb_path)
    cells = nb['cells']
    link_markdown_cells(cells, get_imported_modules(cells, Path(nb_path).stem))
    write_nb(nb, nb_path)
    NotebookNotary().sign(read_nb(nb_path))

def get_module_from_notebook(doc_path):
    "Find module given a source path. Assume it belongs to fastai directory"
    return f'fastai.{Path(doc_path).stem}'

def check_nbconvert_version():
    import nbconvert
    assert nbconvert.version_info >= (5,4,0), "Please update nbconvert to >=5.4 for consistent .html output"

def update_notebooks(source_path, dest_path=None, update_html=True, document_new_fns=False,
                     update_nb_links=True, html_path=None, force=False):
    "`source_path` can be a directory or a file. Assume all modules reside in the fastai directory."
    from .convert2html import convert_nb
    source_path = Path(source_path)

    if source_path.is_file():
        dest_path = source_path.parent if dest_path is None else Path(dest_path)
        html_path = dest_path/'..'/'docs' if html_path is None else Path(html_path)
        doc_path = source_path
        assert source_path.suffix == '.ipynb', 'Must update from notebook or module'
        if document_new_fns:
            mod = import_mod(get_module_from_notebook(source_path))
            if not mod: print('Could not find module for path:', source_path)
            elif mod.__file__.endswith('__init__.py'): pass
            else: update_module_page(mod, dest_path)
        generate_missing_metadata(doc_path)
        if update_nb_links:
            print(f'Updating notebook {doc_path}. Please wait...')
            link_nb(doc_path)
            execute_nb(doc_path, {'metadata': {'path': doc_path.parent}}, show_doc_only=True)
        if update_html:
            check_nbconvert_version()
            html_fn = html_path/doc_path.with_suffix('.html').name
            if not force and html_fn.is_file():
                in_mod  = os.path.getmtime(doc_path)
                out_mod = os.path.getmtime(html_fn)
                if in_mod < out_mod: return
            convert_nb(doc_path, html_path)

    elif (source_path.name.startswith('fastai.')):
        # Do module update
        assert dest_path is not None, 'To update a module, you must specify a destination folder for where notebook resides'
        mod = import_mod(source_path.name)
        if not mod: return print('Could not find module for:', source_path)
        doc_path = Path(dest_path)/(strip_fastai(mod.__name__)+'.ipynb')
        if not doc_path.exists():
            print('Notebook does not exist. Creating:', doc_path)
            create_module_page(mod, dest_path)
        update_notebooks(doc_path, dest_path=dest_path, update_html=update_html, document_new_fns=document_new_fns,
                         update_nb_links=update_nb_links, html_path=html_path)
    elif source_path.is_dir():
        for f in sorted(Path(source_path).glob('*.ipynb')):
            update_notebooks(f, dest_path=dest_path, update_html=update_html, document_new_fns=document_new_fns,
                             update_nb_links=update_nb_links, html_path=html_path)
    else: print('Could not resolve source file:', source_path)
