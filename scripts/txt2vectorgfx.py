"""
using POTRACE as backend cmd line tool for vectorizing SD output
This script will download from

https://potrace.sourceforge.net/#downloading

the windows exetuable (todo: mac, linux support)
Potrace is under GPL, you can download the source from the url above.

If you dont want to download that, please install POTRACE to your 
system manually and assign it to your PATH env variable properly.
"""

PO_URL     = "https://potrace.sourceforge.net/download/1.16/potrace-1.16.win64.zip"
PO_ZIP     = "potrace.zip"
PO_ZIP_EXE = "potrace-1.16.win64/potrace.exe"
PO_EXE     = "scripts/potrace.exe"

# not yet
BASE_PROMPT=",(((lineart))),((low detail)),(simple),high contrast,sharp,2 bit"
BASE_NEGPROMPT="(((text))),((color)),(shading),background,noise,dithering,gradient,detailed,out of frame,ugly,error,Illustration, watermark"

BASE_STEPS=40
BASE_SCALE=10

StyleDict = {
    "Illustration":BASE_PROMPT+",(((vector graphic))),medium detail",
    "Logo":BASE_PROMPT+",(((centered vector graphic logo))),negative space,stencil,trending on dribbble",
    "Drawing":BASE_PROMPT+",(((cartoon graphic))),childrens book,lineart,negative space",
    "Artistic":BASE_PROMPT+",(((artistic monochrome painting))),precise lineart,negative space",
    "Tattoo":BASE_PROMPT+",(((tattoo template, ink on paper))),uniform lighting,lineart,negative space",
    "Gothic":BASE_PROMPT+",(((gothic ink on paper))),H.P. Lovecraft,Arthur Rackham",
    "Anime":BASE_PROMPT+",(((clean ink anime illustration))),Studio Ghibli,Makoto Shinkai,Hayao Miyazaki,Audrey Kawasaki",
    "Cartoon":BASE_PROMPT+",(((clean ink funny comic cartoon illustration)))",
    "Sticker":",(Die-cut sticker, kawaii sticker,contrasting background, illustration minimalism, vector, pastel colors)",
    "Gold Pendant": ",gold dia de los muertos pendant, intricate 2d vector geometric, cutout shape pendant, blueprint frame lines sharp edges, svg vector style, product studio shoot",
    "None - prompt only":""
}

##########################################################################

import os
import pathlib
import subprocess
from PIL import Image

from tkinter import Image, image_types
from zipfile import ZipFile
import requests
import glob
import os.path
from sys import platform

import modules.scripts as scripts
import modules.images as Images
import gradio as gr

from modules.processing import Processed, process_images
from modules.shared import opts

class Script(scripts.Script):
    def title(self):
        return "Text to Vector Graphics"

    def ui(self, is_img2img):
        with gr.Row():
            poUseColor = gr.Radio(list(StyleDict.keys()), label="Visual style", value="Illustration")

        with gr.Row():

            with gr.Column():
                with gr.Row():
                    poDoVector = gr.Checkbox(label="Enable Vectorizing", value=True)
                    poFormat = gr.Dropdown(["svg","pdf"], label="Output format", value="svg")
                    poOpaque = gr.Checkbox(label="White is Opaque", value=True)
                    poTight = gr.Checkbox(label="Cut white margin from input", value=True)
                with gr.Row():
                    poKeepPnm = gr.Checkbox(label="Keep temp images", value=False)
                    poThreshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5)

            with gr.Column():
                    poTransPNG      = gr.Checkbox(label="Transparent PNG",value=True)
                    poTransPNGEps   = gr.Slider(label="Noise Tolerance",minimum=0,maximum=128,value=16)
                    poTransPNGQuant = gr.Slider(label="Quantize",minimum=1,maximum=255,value=16)

        return [poUseColor,poFormat, poOpaque, poTight, poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poDoVector,poTransPNGQuant]

    def run(self, p, poUseColor,poFormat, poOpaque, poTight, poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poDoVector, poTransPNGQuant):
        PO_TO_CALL = self.check_Potrace_install()

        p.do_not_save_grid = True

        # Add the prompt from above
        p.prompt += StyleDict[poUseColor]

# not yet        
#        p.cfg_scale=BASE_SCALE
#        p.steps = BASE_STEPS

        images = []
        proc = process_images(p)
        images += proc.images        

        # unfortunately the concrete file name is nontrivial using increment counter etc, so we have to reverse-guess the last stored images by changetime
        folder = p.outpath_samples

        if opts.save_to_dirs:
            folder = glob.glob(p.outpath_samples+"/*")
            folder = max(folder, key=os.path.getctime)

        files = glob.glob(folder+"/*."+opts.samples_format)
        # latest first
        files = sorted(files, key=os.path.getctime, reverse=True)

        assert len(files) > 0
        assert len(files) >= len(images), "could not find generated image files. Ensure they are stored at all, best if in subdirectory"

        mixedImages = []
        try:
            # vectorize
            for i,img in enumerate(images[::-1]): 
                fullfn = files[i]
                fullfnPath = pathlib.Path(fullfn)
                
                fullofpnm =  fullfnPath.with_suffix('.pnm') #for vectorizing

                fullofTPNG = fullfnPath.with_stem(fullfnPath.stem+ "_T")
                fullofTPNG = fullofTPNG.with_suffix('.png')

                fullof = pathlib.Path(fullfn).with_suffix('.'+poFormat)

                mixedImages.append(img)

                # set transparency to PNG, actually not vector feature, but people need it
                if poTransPNG:
                    self.doTransPNG(poTransPNGEps, mixedImages, img, fullofTPNG, poTransPNGQuant)

                if poDoVector:
                    self.doVector(poFormat, poOpaque, poTight, poKeepPnm, poThreshold, PO_TO_CALL, img, fullofpnm, fullof)

        except (Exception):
            raise Exception("TXT2Vectorgraphics: Execution of Potrace failed, check filesystem, permissions, installation or settings (is image saving on?)")

        return Processed(p, mixedImages, p.seed, proc.info)

    def doVector(self, poFormat, poOpaque, poTight, poKeepPnm, poThreshold, PO_TO_CALL, img, fullofpnm, fullof):
        # for vectorizing
        img.save(fullofpnm)

        args = [PO_TO_CALL,  "-b", poFormat, "-o", fullof, "--blacklevel", format(poThreshold, 'f')]
        if poOpaque: args.append("--opaque")
        if poTight: args.append("--tight")
        args.append(fullofpnm)

        p2 = subprocess.Popen(args)

        if not poKeepPnm:
            p2.wait()
            os.remove(fullofpnm)

    def doTransPNG(self, poTransPNGEps, mixedImages, img, fullofTPNG, poTransPNGQuant):
        #Image.quantize(colors=256, method=None, kmeans=0, palette=None)
        imgQ = img.quantize(colors=poTransPNGQuant, kmeans=0, palette=None)
        histo = imgQ.histogram()

        # get first pixel and assum it is background, best with Sticker style
        if (imgQ):
            bgI = imgQ.getpixel((0,0)) # return pal index
            bg = list(imgQ.palette.colors.keys())[bgI]

        E = poTransPNGEps # tolerance range if noisy

        imgT=imgQ.convert('RGBA')
        datas = imgT.getdata()
        newData = []
        for item in datas:
            if (item[0] > bg[0]-E and item[0] < bg[0]+E) and (item[1] > bg[1]-E and item[1] < bg[1]+E) and (item[2] > bg[2]-E and item[1] < bg[2]+E):
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        imgT.putdata(newData)
        imgT.save(fullofTPNG)
        mixedImages.append(imgQ)
        mixedImages.append(imgT)

    def check_Potrace_install(self) -> str:
        # For Linux, run potrace from installed binary
        if platform == "darwin":
            try:
                # check whether already in PATH 
                checkPath = subprocess.Popen(["potrace","-v"])
                checkPath.wait()
                return "potrace"
            except (Exception):
                raise Exception("Cannot find installed Protrace on Mac. Please run `brew install potrace`")

        elif platform == "linux"or platform == "linux2":
            try:
                # check whether already in PATH 
                checkPath = subprocess.Popen(["potrace","-v"])
                checkPath.wait()
                return "potrace"
            except (Exception):
                raise Exception("Cannot find installed Potrace. Please run `sudo apt install potrace`")

        # prefer local potrace over that from PATH
        elif platform == "win32":
            if not os.path.exists(PO_EXE):
                try:
                    # check whether already in PATH 
                    checkPath = subprocess.Popen(["potrace","-v"])
                    checkPath.wait()
                    return "potrace"

                except (Exception):
                    try:
                        # try to download Potrace and unzip locally into "scripts"
                        if not os.path.exists(PO_ZIP):
                            r = requests.get(PO_URL)
                            with open(PO_ZIP, 'wb') as f:
                                f.write(r.content) 

                        with ZipFile(PO_ZIP, 'r') as zipObj:
                            exe = zipObj.read(PO_ZIP_EXE)
                            with open(PO_EXE,"wb") as e:
                                e.write(exe)
                                zipObj.close()
                                os.remove(PO_ZIP)
                    except:
                        raise Exception("Cannot find and or download/extract Potrace. Provide Potrace in script folder. ")
        return PO_EXE
