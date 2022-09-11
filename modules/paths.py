import argparse
import os
import sys

script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, script_path)

# search for directory of stable diffsuion in following palces
sd_path = None
possible_sd_paths = [os.path.join(script_path, 'repositories/stable-diffusion'), '.', os.path.dirname(script_path)]
for possible_sd_path in possible_sd_paths:
    if os.path.exists(os.path.join(possible_sd_path, 'ldm/models/diffusion/ddpm.py')):
        sd_path = os.path.abspath(possible_sd_path)

assert sd_path is not None, "Couldn't find Stable Diffusion in any of: " + str(possible_sd_paths)

path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion'),
    (os.path.join(sd_path, '../taming-transformers'), 'taming', 'Taming Transformers'),
    (os.path.join(sd_path, '../CodeFormer'), 'inference_codeformer.py', 'CodeFormer'),
    (os.path.join(sd_path, '../BLIP'), 'models/blip.py', 'BLIP'),
]

paths = {}

for d, must_exist, what in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        sys.path.append(d)
        paths[what] = d
