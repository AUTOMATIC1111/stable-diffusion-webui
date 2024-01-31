import os.path, re, nbformat, jupyter_contrib_nbextensions
from nbconvert.preprocessors import Preprocessor
from nbconvert import HTMLExporter
from traitlets.config import Config
from pathlib import Path

__all__ = ['read_nb', 'convert_nb', 'convert_all']

exporter = HTMLExporter(Config())
exporter.exclude_input_prompt=True
exporter.exclude_output_prompt=True
#Loads the template to deal with hidden cells.
exporter.template_file = 'jekyll.tpl'
path = Path(__file__).parent
exporter.template_path.append(str(path))

def read_nb(fname):
    "Read the notebook in `fname`."
    with open(fname,'r') as f: return nbformat.reads(f.read(), as_version=4)

def convert_nb(fname, dest_path='.'):
    "Convert a notebook `fname` to html file in `dest_path`."
    from .gen_notebooks import remove_undoc_cells, remove_code_cell_jupyter_widget_state_elem
    nb = read_nb(fname)
    nb['cells'] = remove_undoc_cells(nb['cells'])
    nb['cells'] = remove_code_cell_jupyter_widget_state_elem(nb['cells'])
    fname = Path(fname).absolute()
    dest_name = fname.with_suffix('.html').name
    meta = nb['metadata']
    meta_jekyll = meta['jekyll'] if 'jekyll' in meta else {'title': fname.with_suffix('').name}
    meta_jekyll['nb_path'] = f'{fname.parent.name}/{fname.name}'
    with open(f'{dest_path}/{dest_name}','w') as f:
        f.write(exporter.from_notebook_node(nb, resources=meta_jekyll)[0])

def convert_all(folder, dest_path='.', force_all=False):
    "Convert modified notebooks in `folder` to html pages in `dest_path`."
    path = Path(folder)

    changed_cnt = 0
    for fname in path.glob("*.ipynb"):
        # only rebuild modified files
        fname_out = Path(dest_path)/fname.with_suffix('.html').name
        if not force_all and fname_out.exists():
            in_mod  = os.path.getmtime(fname)
            out_mod = os.path.getmtime(fname_out)
            if in_mod < out_mod: continue

        print(f"converting: {fname} => {fname_out}")
        changed_cnt += 1
        convert_nb(fname, dest_path=dest_path)
    if not changed_cnt: print("No notebooks were modified")
