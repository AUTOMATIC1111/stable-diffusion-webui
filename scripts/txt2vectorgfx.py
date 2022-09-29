from inspect import _void
import os
import pathlib
import subprocess

import modules.scripts as scripts
import modules.images as Images
import gradio as gr

from modules.processing import Processed, process_images
from modules.shared import opts


class Script(scripts.Script):
    def title(self):
        return "Text to Vectorgraphics (svg,pdf)"

    def ui(self, is_img2img):
        poFormat = gr.Dropdown(["svg","pdf"],value="svg")
        poOpaque = gr.Checkbox(label="White is Opaque", value=True)
        poTight = gr.Checkbox(label="Cut white margin from input", value=True)
        poKeepPnm = gr.Checkbox(label="Keep temp images", value=False)
        poThreshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5)

        return [poFormat,poOpaque, poTight, poKeepPnm, poThreshold]

    def run(self, p, poFormat, poOpaque, poTight, poKeepPnm, poThreshold):
        p.do_not_save_grid = True

        images = []
        proc = process_images(p)
        images += proc.images

        # vectorize
        for i,img in enumerate(images): 
            fullfn = Images.save_image(img, p.outpath_samples, "", p.seed, p.prompt, "pnm" )
            fullof = pathlib.Path(fullfn).with_suffix('.'+poFormat)

            args = [opts.potrace_path,  "-b", poFormat, "-o", fullof, "--blacklevel",  format(poThreshold, 'f')]
            if poOpaque: args.append("--opaque")
            if poTight: args.append("--tight")
            args.append(fullfn)

            p2 = subprocess.Popen(args)

            if not poKeepPnm:
                p2.wait()
                os.remove(fullfn)

        return Processed(p, images, p.seed, "")

"""	
potrace 1.16. Transforms bitmaps into vector graphics.

Usage: potrace [options] [filename...]
General options:
 -h, --help                 - print this help message and exit
 -v, --version              - print version info and exit
 -l, --license              - print license info and exit
File selection:
 filename                   - an input file
 -o, --output filename      - write all output to this file
 --                         - end of options; 0 or more input filenames follow
Backend selection:
 -b, --backend name         - select backend by name
 -b svg, -s, --svg          - SVG backend (scalable vector graphics)
 -b pdf                     - PDF backend (portable document format)
 -b pdfpage                 - fixed page-size PDF backend
 -b eps, -e, --eps          - EPS backend (encapsulated PostScript) (default)
 -b ps, -p, --postscript    - PostScript backend
 -b pgm, -g, --pgm          - PGM backend (portable greymap)
 -b dxf                     - DXF backend (drawing interchange format)
 -b geojson                 - GeoJSON backend
 -b gimppath                - Gimppath backend (GNU Gimp)
 -b xfig                    - XFig backend
Algorithm options:
 -z, --turnpolicy policy    - how to resolve ambiguities in path decomposition
 -t, --turdsize n           - suppress speckles of up to this size (default 2)
 -a, --alphamax n           - corner threshold parameter (default 1)
 -n, --longcurve            - turn off curve optimization
 -O, --opttolerance n       - curve optimization tolerance (default 0.2)
 -u, --unit n               - quantize output to 1/unit pixels (default 10)
 -d, --debug n              - produce debugging output of type n (n=1,2,3)
Scaling and placement options:
 -P, --pagesize format      - page size (default is letter)
 -W, --width dim            - width of output image
 -H, --height dim           - height of output image
 -r, --resolution n[xn]     - resolution (in dpi) (dimension-based backends)
 -x, --scale n[xn]          - scaling factor (pixel-based backends)
 -S, --stretch n            - yresolution/xresolution
 -A, --rotate angle         - rotate counterclockwise by angle
 -M, --margin dim           - margin
 -L, --leftmargin dim       - left margin
 -R, --rightmargin dim      - right margin
 -T, --topmargin dim        - top margin
 -B, --bottommargin dim     - bottom margin
 --tight                    - remove whitespace around the input image
Color options, supported by some backends:
 -C, --color #rrggbb        - set foreground color (default black)
 --fillcolor #rrggbb        - set fill color (default transparent)
 --opaque                   - make white shapes opaque
SVG options:
 --group                    - group related paths together
 --flat                     - whole image as a single path
Postscript/EPS/PDF options:
 -c, --cleartext            - do not compress the output
 -2, --level2               - use postscript level 2 compression (default)
 -3, --level3               - use postscript level 3 compression
 -q, --longcoding           - do not optimize for file size
PGM options:
 -G, --gamma n              - gamma value for anti-aliasing (default 2.2)
Frontend options:
 -k, --blacklevel n         - black/white cutoff in input file (default 0.5)
 -i, --invert               - invert bitmap
Progress bar options:
 --progress                 - show progress bar
 --tty mode                 - progress bar rendering: vt100 or dumb

Dimensions can have optional units, e.g. 6.5in, 15cm, 100pt.
Default is inches (or pixels for pgm, dxf, and gimppath backends).
Possible input file formats are: pnm (pbm, pgm, ppm), bmp.
Backends are: svg, pdf, pdfpage, eps, postscript, ps, dxf, geojson, pgm, 
gimppath, xfig.
"""