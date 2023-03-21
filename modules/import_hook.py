import sys
import os

models_path = "models"
# this will break any attempt to import xformers which will prevent stability diffusion repo from trying to use it
if "--xformers" not in "".join(sys.argv):
    sys.modules["xformers"] = None

if "--no-half-vae" not in "".join(sys.argv):
    sys.argv.append('--no-half-vae')
sys.argv.append(f'--clip-models-path={models_path}/CLIP')
