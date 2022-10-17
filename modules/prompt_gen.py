# We need this so Python doesn't complain about the unknown StableDiffusionProcessing-typehint at runtime
from __future__ import annotations

import os
import os.path

def load_promptgen_text(path: str):
    if os.path.exists(path):
        promptgen_dict = {}
        with open(path, "r", encoding="utf8") as file:
            count = 1
            for line in file:
 
                value = line.split()
 
                promptgen_dict[int(count)] = value
                count += 1
    return promptgen_dict
