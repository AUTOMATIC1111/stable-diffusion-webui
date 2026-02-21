import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401

import modules.safe  # noqa: F401


def mute_sdxl_imports():
    """create fake modules that SDXL wants to import but doesn't actually use for our purposes"""

    class Dummy:
        pass

    module = Dummy()
    module.LPIPS = None
    sys.modules['taming.modules.losses.lpips'] = module

    module = Dummy()
    module.StableDataModuleFromConfig = None
    sys.modules['sgm.data'] = module

    # Mock taming.modules.vqvae.quantize for VQModel/VQModelInterface
    class DummyVectorQuantizer:
        def __init__(self, *args, **kwargs):
            pass
    module = Dummy()
    module.VectorQuantizer2 = DummyVectorQuantizer
    sys.modules['taming.modules.vqvae.quantize'] = module

    # Mock taming.modules.vqvae for completeness
    module = Dummy()
    sys.modules['taming.modules.vqvae'] = module

    # Mock taming.modules for completeness
    module = Dummy()
    sys.modules['taming.modules'] = module

    # Mock ldm.modules.midas for depth estimation
    class MidasApiDummy:
        ISL_PATHS = {}
        load_model = None
        load_model_inner = None
    module = Dummy()
    module.api = MidasApiDummy()
    sys.modules['ldm.modules.midas'] = module

    # Mock sgm module for SDXL when generative-models repo is not available
    import types
    
    def create_sgm_modules():
        """Create all required sgm module mocks"""
        sgm_module = types.ModuleType('sgm')
        sgm_models = types.ModuleType('sgm.models')
        sgm_diffusion = types.ModuleType('sgm.models.diffusion')
        sgm_sgm = types.ModuleType('sgm.models.diffusion.sgm')
        sgm_modules = types.ModuleType('sgm.modules')
        sgm_diffusionmodules = types.ModuleType('sgm.modules.diffusionmodules')
        sgm_attention = types.ModuleType('sgm.modules.attention')
        sgm_encoders = types.ModuleType('sgm.modules.encoders')
        sgm_encoders_modules = types.ModuleType('sgm.modules.encoders.modules')
        
        # Add submodules with proper attributes
        class DummyDiffusionEngine:
            pass
        sgm_diffusion.DiffusionEngine = DummyDiffusionEngine
        
        class DummyDenoiserScaling:
            pass
        sgm_diffusionmodules.denoiser_scaling = DummyDenoiserScaling()
        
        class DummyDiscretizer:
            pass
        sgm_diffusionmodules.discretizer = DummyDiscretizer()
        
        class DummyAttention:
            XFORMERS_IS_AVAILABLE = False
            SDP_IS_AVAILABLE = True
        sgm_attention.CrossAttention = DummyAttention
        
        class DummyModelClass:
            class AttnBlock:
                forward = lambda self, *args, **kwargs: None
            nonlinearity = lambda *args, **kwargs: None
        sgm_diffusionmodules.model = DummyModelClass()
        
        class DummyUNetModel:
            forward = lambda self, *args, **kwargs: None
        
        class DummyOpenAIModelClass:
            UNetModel = DummyUNetModel
        sgm_diffusionmodules.openaimodel = DummyOpenAIModelClass()
        
        class DummyGeneralConditioner:
            pass
        sgm_modules.GeneralConditioner = DummyGeneralConditioner
        
        class DummyUtil:
            pass
        sgm_diffusionmodules.util = DummyUtil()
        
        # Set up module hierarchy - this is critical for attribute access
        sgm_module.models = sgm_models
        sgm_module.modules = sgm_modules
        sgm_models.diffusion = sgm_diffusion
        sgm_diffusion.sgm = sgm_sgm
        sgm_modules.diffusionmodules = sgm_diffusionmodules
        sgm_modules.attention = sgm_attention
        sgm_modules.encoders = sgm_encoders
        sgm_encoders.modules = sgm_encoders_modules
        
        # Register all modules
        sys.modules['sgm'] = sgm_module
        sys.modules['sgm.models'] = sgm_models
        sys.modules['sgm.models.diffusion'] = sgm_diffusion
        sys.modules['sgm.models.diffusion.sgm'] = sgm_sgm
        sys.modules['sgm.modules'] = sgm_modules
        sys.modules['sgm.modules.diffusionmodules'] = sgm_diffusionmodules
        sys.modules['sgm.modules.attention'] = sgm_attention
        sys.modules['sgm.modules.encoders'] = sgm_encoders
        sys.modules['sgm.modules.encoders.modules'] = sgm_encoders_modules
        sys.modules['sgm.modules.diffusionmodules.denoiser_scaling'] = sgm_diffusionmodules.denoiser_scaling
        sys.modules['sgm.modules.diffusionmodules.discretizer'] = sgm_diffusionmodules.discretizer
        sys.modules['sgm.modules.diffusionmodules.model'] = sgm_diffusionmodules.model
        sys.modules['sgm.modules.diffusionmodules.openaimodel'] = sgm_diffusionmodules.openaimodel
        sys.modules['sgm.modules.diffusionmodules.util'] = sgm_diffusionmodules.util
        
    create_sgm_modules()


# data_path = cmd_opts_pre.data
sys.path.insert(0, script_path)

# search for directory of stable diffusion in following places
sd_path = None
possible_sd_paths = [os.path.join(script_path, 'repositories/stable-diffusion-stability-ai'), '.', os.path.dirname(script_path)]
for possible_sd_path in possible_sd_paths:
    if os.path.exists(os.path.join(possible_sd_path, 'ldm/models/diffusion/ddpm.py')):
        sd_path = os.path.abspath(possible_sd_path)
        break

assert sd_path is not None, f"Couldn't find Stable Diffusion in any of: {possible_sd_paths}"

mute_sdxl_imports()

path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion', []),
    (os.path.join(sd_path, '../generative-models'), 'sgm', 'Stable Diffusion XL', ["sgm"]),
    (os.path.join(sd_path, '../BLIP'), 'models/blip.py', 'BLIP', []),
    (os.path.join(sd_path, '../k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
]

paths = {}

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
        # Add fallback path for SDXL even if not found to prevent KeyError
        if what == "Stable Diffusion XL":
            paths[what] = os.path.join(script_path, 'repositories/generative-models')
    else:
        d = os.path.abspath(d)
        if "atstart" in options:
            sys.path.insert(0, d)
        elif "sgm" in options:
            # Stable Diffusion XL repo has scripts dir with __init__.py in it which ruins every extension's scripts dir, so we
            # import sgm and remove it from sys.path so that when a script imports scripts.something, it doesbn't use sgm's scripts dir.

            sys.path.insert(0, d)
            import sgm  # noqa: F401
            sys.path.pop(0)
        else:
            sys.path.append(d)
        paths[what] = d
