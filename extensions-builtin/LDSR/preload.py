import os
from modules import paths


def preload(parser):
    parser.add_argument("--ldsr-models-path", type=str, help="Path to directory with LDSR model file(s).", default=os.path.join(paths.models_path, 'LDSR'))
