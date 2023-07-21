import os
from modules import paths


def preload(parser):
    parser.add_argument("--lora-dir", type=str, help="Path to directory with Lora networks.", default=os.path.join(paths.models_path, 'Lora'))
    parser.add_argument("--lyco-dir-backcompat", type=str, help="Path to directory with LyCORIS networks (for backawards compatibility; can also use --lyco-dir).", default=os.path.join(paths.models_path, 'LyCORIS'))
