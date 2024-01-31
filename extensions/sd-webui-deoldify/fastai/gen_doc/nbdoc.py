"`gen_doc.nbdoc` generates notebook documentation from module functions and links to correct places"

import inspect,importlib,enum,os,re,nbconvert
from IPython.core.display import display, Markdown, HTML
from nbconvert import HTMLExporter
from IPython.core import page
from IPython import get_ipython
from typing import Dict, Any, AnyStr, List, Sequence, TypeVar, Tuple, Optional, Union
from .docstrings import *
from .core import *
from ..torch_core import *
from .nbtest import get_pytest_html
from ..utils.ipython import IS_IN_COLAB

__all__ = ['get_fn_link', 'link_docstring', 'show_doc', 'get_ft_names', 'md2html',
           'get_exports', 'show_video', 'show_video_from_youtube', 'import_mod', 'get_source_link',
           'is_enum', 'jekyll_note', 'jekyll_warn', 'jekyll_important', 'doc']

MODULE_NAME = 'fastai'
SOURCE_URL = 'https://github.com/fastai/fastai/blob/master/'
PYTORCH_DOCS = 'https://pytorch.org/docs/stable/'
FASTAI_DOCS = 'https://docs.fast.ai'
use_relative_links = True

_typing_names = {t:n for t,n in fastai_types.items() if t.__module__=='typing'}
arg_prefixes = {inspect._VAR_POSITIONAL: '\*', inspect._VAR_KEYWORD:'\*\*'}


def is_enum(cls): return cls == enum.Enum or cls == enum.EnumMeta

def link_type(arg_type, arg_name=None, include_bt:bool=True):
    "Create link to documentation."
    arg_name = arg_name or fn_name(arg_type)
    if include_bt: arg_name = code_esc(arg_name)
    if belongs_to_module(arg_type, 'torch') and ('Tensor' not in arg_name): return f'[{arg_name}]({get_pytorch_link(arg_type)})'
    if is_fastai_class(arg_type): return f'[{arg_name}]({get_fn_link(arg_type)})'
    return arg_name

def is_fastai_class(t): return belongs_to_module(t, MODULE_NAME)

def belongs_to_module(t, module_name):
    "Check if `t` belongs to `module_name`."
    if hasattr(t, '__func__'): return belongs_to_module(t.__func__, module_name)
    if not inspect.getmodule(t): return False
    return inspect.getmodule(t).__name__.startswith(module_name)

def code_esc(s): return f'`{s}`'

def type_repr(t):
    if t in _typing_names: return link_type(t, _typing_names[t])
    if isinstance(t, partial): return partial_repr(t)
    if hasattr(t, '__forward_arg__'): return link_type(t.__forward_arg__)
    elif getattr(t, '__args__', None):
        args = t.__args__
        if len(args)==2 and args[1] == type(None):
            return f'`Optional`\[{type_repr(args[0])}\]'
        reprs = ', '.join([type_repr(o) for o in args])
        return f'{link_type(t)}\[{reprs}\]'
    else: return link_type(t)

def partial_repr(t):
    args = (t.func,) + t.args + tuple([f'{k}={v}' for k,v in t.keywords.items()])
    reprs = ', '.join([link_type(o) for o in args])
    return f'<code>partial(</code>{reprs}<code>)</code>'

def anno_repr(a): return type_repr(a)

def format_param(p):
    "Formats function param to `param1:Type=val`. Font weights: param1=bold, val=bold+italic"
    arg_prefix = arg_prefixes.get(p.kind, '') # asterisk prefix for *args and **kwargs
    res = f"**{arg_prefix}{code_esc(p.name)}**"
    if hasattr(p, 'annotation') and p.annotation != p.empty: res += f':{anno_repr(p.annotation)}'
    if p.default != p.empty:
        default = getattr(p.default, 'func', p.default)
        default = getattr(default, '__name__', default)
        res += f'=***`{repr(default)}`***'
    return res

def format_ft_def(func, full_name:str=None)->str:
    "Format and link `func` definition to show in documentation"
    sig = inspect.signature(func)
    name = f'<code>{full_name or func.__name__}</code>'
    fmt_params = [format_param(param) for name,param
                  in sig.parameters.items() if name not in ('self','cls')]
    arg_str = f"({', '.join(fmt_params)})"
    if sig.return_annotation and (sig.return_annotation != sig.empty): arg_str += f" → {anno_repr(sig.return_annotation)}"
    if is_fastai_class(type(func)):        arg_str += f" :: {link_type(type(func))}"
    f_name = f"<code>class</code> {name}" if inspect.isclass(func) else name
    return f'{f_name}',f'{name}{arg_str}'

def get_enum_doc(elt, full_name:str)->str:
    "Formatted enum documentation."
    vals = ', '.join(elt.__members__.keys())
    return f'{code_esc(full_name)}',f'<code>Enum</code> = [{vals}]'

def get_cls_doc(elt, full_name:str)->str:
    "Class definition."
    parent_class = inspect.getclasstree([elt])[-1][0][1][0]
    name,args = format_ft_def(elt, full_name)
    if parent_class != object: args += f' :: {link_type(parent_class, include_bt=True)}'
    return name,args

def show_doc(elt, doc_string:bool=True, full_name:str=None, arg_comments:dict=None, title_level=None, alt_doc_string:str='',
             ignore_warn:bool=False, markdown=True, show_tests=True):
    "Show documentation for element `elt`. Supported types: class, Callable, and enum."
    arg_comments = ifnone(arg_comments, {})
    anchor_id = get_anchor(elt)
    elt = getattr(elt, '__func__', elt)
    full_name = full_name or fn_name(elt)
    if inspect.isclass(elt):
        if is_enum(elt.__class__):   name,args = get_enum_doc(elt, full_name)
        else:                        name,args = get_cls_doc(elt, full_name)
    elif isinstance(elt, Callable):  name,args = format_ft_def(elt, full_name)
    else: raise Exception(f'doc definition not supported for {full_name}')
    source_link = get_function_source(elt) if is_fastai_class(elt) else ""
    test_link, test_modal = get_pytest_html(elt, anchor_id=anchor_id) if show_tests else ('', '')
    title_level = ifnone(title_level, 2 if inspect.isclass(elt) else 4)
    doc =  f'<h{title_level} id="{anchor_id}" class="doc_header">{name}{source_link}{test_link}</h{title_level}>'
    doc += f'\n\n> {args}\n\n'
    doc += f'{test_modal}'
    if doc_string and (inspect.getdoc(elt) or arg_comments):
        doc += format_docstring(elt, arg_comments, alt_doc_string, ignore_warn) + ' '
    if markdown: display(Markdown(doc))
    else: return doc

def md2html(md):
    if nbconvert.__version__ < '5.5.0': return HTMLExporter().markdown2html(md)
    else: return HTMLExporter().markdown2html(defaultdict(lambda: defaultdict(dict)), md)
    
def doc(elt):
    "Show `show_doc` info in preview window along with link to full docs."
    global use_relative_links
    use_relative_links = False
    elt = getattr(elt, '__func__', elt)
    md = show_doc(elt, markdown=False)
    if is_fastai_class(elt):
        md += f'\n\n<a href="{get_fn_link(elt)}" target="_blank" rel="noreferrer noopener">Show in docs</a>'
    output = md2html(md)
    use_relative_links = True
    if IS_IN_COLAB: get_ipython().run_cell_magic(u'html', u'', output)
    else:
        try: page.page({'text/html': output})
        except: display(Markdown(md))

def format_docstring(elt, arg_comments:dict={}, alt_doc_string:str='', ignore_warn:bool=False)->str:
    "Merge and format the docstring definition with `arg_comments` and `alt_doc_string`."
    parsed = ""
    doc = parse_docstring(inspect.getdoc(elt))
    description = alt_doc_string or f"{doc['short_description']} {doc['long_description']}"
    if description: parsed += f'\n\n{link_docstring(inspect.getmodule(elt), description)}'

    resolved_comments = {**doc.get('comments', {}), **arg_comments} # arg_comments takes priority
    args = inspect.getfullargspec(elt).args if not is_enum(elt.__class__) else elt.__members__.keys()
    if resolved_comments: parsed += '\n'
    for a in resolved_comments:
        parsed += f'\n- *{a}*: {resolved_comments[a]}'
        if a not in args and not ignore_warn: warn(f'Doc arg mismatch: {a}')

    return_comment = arg_comments.get('return') or doc.get('return')
    if return_comment: parsed += f'\n\n*return*: {return_comment}'
    return parsed

_modvars = {}

def replace_link(m):
    keyword = m.group(1) or m.group(2)
    elt = find_elt(_modvars, keyword)
    if elt is None: return m.group()
    return link_type(elt, arg_name=keyword)

# Finds all places with a backtick but only if it hasn't already been linked
BT_REGEX = re.compile("\[`([^`]*)`\](?:\([^)]*\))|`([^`]*)`") # matches [`key`](link) or `key`
def link_docstring(modules, docstring:str, overwrite:bool=False)->str:
    "Search `docstring` for backticks and attempt to link those functions to respective documentation."
    mods = listify(modules)
    for mod in mods: _modvars.update(mod.__dict__) # concat all module definitions
    return re.sub(BT_REGEX, replace_link, docstring)

def find_elt(modvars, keyword, match_last=False):
    "Attempt to resolve keywords such as Learner.lr_find. `match_last` starts matching from last component."
    keyword = strip_fastai(keyword)
    if keyword in modvars: return modvars[keyword]
    comps = keyword.split('.')
    comp_elt = modvars.get(comps[0])
    if hasattr(comp_elt, '__dict__'): return find_elt(comp_elt.__dict__, '.'.join(comps[1:]), match_last=match_last)

def import_mod(mod_name:str, ignore_errors=False):
    "Return module from `mod_name`."
    splits = str.split(mod_name, '.')
    try:
        if len(splits) > 1 : mod = importlib.import_module('.' + '.'.join(splits[1:]), splits[0])
        else: mod = importlib.import_module(mod_name)
        return mod
    except:
        if not ignore_errors: print(f"Module {mod_name} doesn't exist.")

def show_doc_from_name(mod_name, ft_name:str, doc_string:bool=True, arg_comments:dict={}, alt_doc_string:str=''):
    "Show documentation for `ft_name`, see `show_doc`."
    mod = import_mod(mod_name)
    splits = str.split(ft_name, '.')
    assert hasattr(mod, splits[0]), print(f"Module {mod_name} doesn't have a function named {splits[0]}.")
    elt = getattr(mod, splits[0])
    for i,split in enumerate(splits[1:]):
        assert hasattr(elt, split), print(f"Class {'.'.join(splits[:i+1])} doesn't have a function named {split}.")
        elt = getattr(elt, split)
    show_doc(elt, doc_string, ft_name, arg_comments, alt_doc_string)

def get_exports(mod):
    public_names = mod.__all__ if hasattr(mod, '__all__') else dir(mod)
    #public_names.sort(key=str.lower)
    return [o for o in public_names if not o.startswith('_')]

def get_ft_names(mod, include_inner=False)->List[str]:
    "Return all the functions of module `mod`."
    # If the module has an attribute __all__, it picks those.
    # Otherwise, it returns all the functions defined inside a module.
    fn_names = []
    for elt_name in get_exports(mod):
        elt = getattr(mod,elt_name)
        #This removes the files imported from elsewhere
        try:    fname = inspect.getfile(elt)
        except: continue
        if mod.__file__.endswith('__init__.py'):
            if inspect.ismodule(elt): fn_names.append(elt_name)
            else: continue
        else:
            if (fname != mod.__file__): continue
            if inspect.isclass(elt) or inspect.isfunction(elt): fn_names.append(elt_name)
            else: continue
        if include_inner and inspect.isclass(elt) and not is_enum(elt.__class__):
            fn_names.extend(get_inner_fts(elt))
    return fn_names

def get_inner_fts(elt)->List[str]:
    "List the inner functions of a class."
    fts = []
    for ft_name in elt.__dict__.keys():
        if ft_name.startswith('_'): continue
        ft = getattr(elt, ft_name)
        if inspect.isfunction(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.ismethod(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.isclass(ft): fts += [f'{elt.__name__}.{n}' for n in get_inner_fts(ft)]
    return fts

def get_module_toc(mod_name):
    "Display table of contents for given `mod_name`."
    mod = import_mod(mod_name)
    ft_names = mod.__all__ if hasattr(mod,'__all__') else get_ft_names(mod)
    ft_names.sort(key = str.lower)
    tabmat = ''
    for ft_name in ft_names:
        tabmat += f'- [{ft_name}](#{ft_name})\n'
        elt = getattr(mod, ft_name)
        if inspect.isclass(elt) and not is_enum(elt.__class__):
            in_ft_names = get_inner_fts(elt)
            for name in in_ft_names:
                tabmat += f'  - [{name}](#{name})\n'
    display(Markdown(tabmat))

def show_video(url):
    "Display video in `url`."
    data = f'<iframe width="560" height="315" src="{url}" frameborder="0" allowfullscreen></iframe>'
    return display(HTML(data))

def show_video_from_youtube(code, start=0):
    "Display video from Youtube with a `code` and a `start` time."
    url = f'https://www.youtube.com/embed/{code}?start={start}&amp;rel=0&amp;controls=0&amp;showinfo=0'
    return show_video(url)

def get_anchor(fn)->str:
    if hasattr(fn,'__qualname__'): return fn.__qualname__
    if inspect.ismethod(fn): return fn_name(fn.__self__) + '.' + fn_name(fn)
    return fn_name(fn)

def fn_name(ft)->str:
    if ft.__hash__ and ft in _typing_names: return _typing_names[ft]
    if hasattr(ft, '__name__'):   return ft.__name__
    elif hasattr(ft,'_name') and ft._name: return ft._name
    elif hasattr(ft,'__origin__'): return str(ft.__origin__).split('.')[-1]
    else:                          return str(ft).split('.')[-1]

def get_fn_link(ft)->str:
    "Return function link to notebook documentation of `ft`. Private functions link to source code"
    ft = getattr(ft, '__func__', ft)
    anchor = strip_fastai(get_anchor(ft))
    module_name = strip_fastai(get_module_name(ft))
    base = '' if use_relative_links else FASTAI_DOCS
    return f'{base}/{module_name}.html#{anchor}'

def get_module_name(ft)->str: return inspect.getmodule(ft).__name__

def get_pytorch_link(ft)->str:
    "Returns link to pytorch docs of `ft`."
    name = ft.__name__
    ext = '.html'
    if name == 'device': return f'{PYTORCH_DOCS}tensor_attributes{ext}#torch-device'
    if name == 'Tensor': return f'{PYTORCH_DOCS}tensors{ext}#torch-tensor'
    if name.startswith('torchvision'):
        doc_path = get_module_name(ft).replace('.', '/')
        if inspect.ismodule(ft): name = name.replace('.', '-')
        return f'{PYTORCH_DOCS}{doc_path}{ext}#{name}'
    if name.startswith('torch.nn') and inspect.ismodule(ft): # nn.functional is special case
        nn_link = name.replace('.', '-')
        return f'{PYTORCH_DOCS}nn{ext}#{nn_link}'
    paths = get_module_name(ft).split('.')
    if len(paths) == 1: return f'{PYTORCH_DOCS}{paths[0]}{ext}#{paths[0]}.{name}'

    offset = 1 if paths[1] == 'utils' else 0 # utils is a pytorch special case
    doc_path = paths[1+offset]
    if inspect.ismodule(ft): return f'{PYTORCH_DOCS}{doc_path}{ext}#module-{name}'
    fnlink = '.'.join(paths[:(2+offset)]+[name])
    return f'{PYTORCH_DOCS}{doc_path}{ext}#{fnlink}'

def get_source_link(file, line, display_text="[source]", **kwargs)->str:
    "Returns github link for given file"
    link = f"{SOURCE_URL}{file}#L{line}"
    if display_text is None: return link
    return f'<a href="{link}" class="source_link" style="float:right">{display_text}</a>'

def get_function_source(ft, **kwargs)->str:
    "Returns link to `ft` in source code."
    try: line = inspect.getsourcelines(ft)[1]
    except Exception: return ''
    mod_path = get_module_name(ft).replace('.', '/') + '.py'
    return get_source_link(mod_path, line, **kwargs)

def title_md(s:str, title_level:int, markdown=True):
    res = '#' * title_level
    if title_level: res += ' '
    return Markdown(res+s) if markdown else (res+s)

def jekyll_div(s,c,h,icon=None):
    icon = ifnone(icon,c)
    res = f'<div markdown="span" class="alert alert-{c}" role="alert"><i class="fa fa-{c}-circle"></i> <b>{h}: </b>{s}</div>'
    display(Markdown(res))

def jekyll_note(s): return jekyll_div(s,'info','Note')
def jekyll_warn(s): return jekyll_div(s,'danger','Warning', 'exclamation')
def jekyll_important(s): return jekyll_div(s,'warning','Important')
