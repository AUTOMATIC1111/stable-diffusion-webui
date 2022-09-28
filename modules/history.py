import os
import re

from modules import processing, shared, images, devices
from modules.shared import opts
from modules.ui import plaintext_to_html

def history(query, count):
    imagedir = opts.outdir_samples or opts.outdir_txt2img_samples
    images = [os.path.join(imagedir, f)
              for f in os.listdir(imagedir)
              if f.endswith(".png") and re.search(query, f)]
    images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return images[:count], plaintext_to_html("")
