import inspect
import re
from modules.merging import merge_methods
from modules.merging.merge_presets import BLOCK_WEIGHTS_PRESETS, SDXL_BLOCK_WEIGHTS_PRESETS

ALL_PRESETS = {}
ALL_PRESETS.update(BLOCK_WEIGHTS_PRESETS)
ALL_PRESETS.update(SDXL_BLOCK_WEIGHTS_PRESETS)

MERGE_METHODS = dict(inspect.getmembers(merge_methods, inspect.isfunction))
BETA_METHODS = [
    name
    for name, fn in MERGE_METHODS.items()
    if "beta" in inspect.getfullargspec(fn)[0]
]
TRIPLE_METHODS = [
    name
    for name, fn in MERGE_METHODS.items()
    if "c" in inspect.getfullargspec(fn)[0]
]


def interpolate(values, interp_lambda):
    interpolated = []
    for i in range(len(values[0])):
        interpolated.append((1 - interp_lambda) * values[0][i] + interp_lambda * values[1][i])
    return interpolated


class WeightClass:
    def __init__(self,
                 model_a,
                 **kwargs,
                 ):
        self.SDXL = "model.diffusion_model.middle_block.1.transformer_blocks.9.norm3.weight" in model_a.keys()
        self.NUM_INPUT_BLOCKS = 12 if not self.SDXL else 9
        self.NUM_MID_BLOCK = 1
        self.NUM_OUTPUT_BLOCKS = 12 if not self.SDXL else 9
        self.NUM_TOTAL_BLOCKS = self.NUM_INPUT_BLOCKS + self.NUM_MID_BLOCK + self.NUM_OUTPUT_BLOCKS
        self.iterations = kwargs.get("re_basin_iterations", 1)
        self.it = 0
        self.re_basin = kwargs.get("re_basin", False)
        self.ratioDict = {}
        for key, value in kwargs.items():
            if isinstance(value, list) or (key.lower() not in ["alpha", "beta"]):
                self.ratioDict[key.lower()] = value
            else:
                self.ratioDict[key.lower()] = [value]

        for key, value in self.ratioDict.items():
            if key in ["alpha", "beta"]:
                for i, v in enumerate(value):
                    if isinstance(v, str) and v.upper() in BLOCK_WEIGHTS_PRESETS.keys():
                        value[i] = BLOCK_WEIGHTS_PRESETS[v.upper()]
                    else:
                        value[i] = [float(x) for x in v.split(",")] if isinstance(v, str) else v
                        if not isinstance(value[i], list):
                            value[i] = [value[i]] * (self.NUM_TOTAL_BLOCKS + 1)
                if len(value) > 1 and isinstance(value[0], list):
                    self.ratioDict[key] = interpolate(value, self.ratioDict.get(key + "_lambda", 0))
                else:
                    self.ratioDict[key] = self.ratioDict[key][0]

    def __call__(self, key, it=0):
        current_bases = {}
        if "alpha" in self.ratioDict:
            current_bases["alpha"] = self.step_weights_and_bases(self.ratioDict["alpha"])
        if "beta" in self.ratioDict:
            current_bases["beta"] = self.step_weights_and_bases(self.ratioDict["beta"])

        weight_index = 0
        if "model" in key:

            if "model.diffusion_model." in key:
                weight_index = -1

                re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
                re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
                re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12

                if "time_embed" in key:
                    weight_index = 0  # before input blocks
                elif ".out." in key:
                    weight_index = self.NUM_TOTAL_BLOCKS - 1  # after output blocks
                elif m := re_inp.search(key):
                    weight_index = int(m.groups()[0])
                elif re_mid.search(key):
                    weight_index = self.NUM_INPUT_BLOCKS
                elif m := re_out.search(key):
                    weight_index = self.NUM_INPUT_BLOCKS + self.NUM_MID_BLOCK + int(m.groups()[0])

                if weight_index >= self.NUM_TOTAL_BLOCKS:
                    raise ValueError(f"illegal block index {key}")

        current_bases = {k: w[weight_index] for k, w in current_bases.items()}
        return current_bases

    def step_weights_and_bases(self, ratio):
        if not self.re_basin:
            return ratio

        new_ratio = [
            1 - (1 - (1 + self.it) * v / self.iterations) / (1 - self.it * v / self.iterations)
            if self.it > 0
            else v / self.iterations
            for v in ratio
        ]
        return new_ratio

    def set_it(self, it):
        self.it = it
