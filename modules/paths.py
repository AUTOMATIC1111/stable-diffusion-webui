import argparse
import os
import sys

script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, script_path)

# use current directory as SD dir if it has related files, otherwise parent dir of script as stated in guide
sd_path = os.path.abspath('.') if os.path.exists('./ldm/models/diffusion/ddpm.py') else os.path.dirname(script_path)

# add parent directory to path; this is where Stable diffusion repo should be
path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion'),
    (os.path.join(sd_path, '../taming-transformers'), 'taming', 'Taming Transformers')
]
for d, must_exist, what in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        sys.path.append(os.path.join(script_path, d))
