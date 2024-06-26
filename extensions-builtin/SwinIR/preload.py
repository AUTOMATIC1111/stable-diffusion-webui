import os
from modules import paths


def preload(parser):
    parser.add_argument("--swinir-models-path", type=str, help="Path to directory with SwinIR model file(s).", default=os.path.join(paths.models_path, 'SwinIR'))
