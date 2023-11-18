import time
import logging
import torch
from modules import shared, devices
from installer import setup_logging


#Used by OpenVINO, can be used with TensorRT or Olive
class CompiledModelState:
    def __init__(self):
        self.first_pass = True
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.partition_id = 0
        self.cn_model = []
        self.lora_model = []
        self.lora_compile = False
        self.compiled_cache = {}
        self.partitioned_modules = {}


def optimize_ipex(sd_model):
    try:
        t0 = time.time()
        import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
        sd_model.unet.training = False
        sd_model.unet = ipex.optimize(sd_model.unet, dtype=devices.dtype_unet, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
        if hasattr(sd_model, 'vae'):
            sd_model.vae.training = False
            sd_model.vae = ipex.optimize(sd_model.vae, dtype=devices.dtype_vae, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
        if hasattr(sd_model, 'movq'):
            sd_model.movq.training = False
            sd_model.movq = ipex.optimize(sd_model.movq, dtype=devices.dtype_vae, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
        t1 = time.time()
        shared.log.info(f"Model compile: mode=IPEX-optimize time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Model compile: task=IPEX-optimize error: {e}")


def optimize_openvino():
    try:
        from modules.intel.openvino import openvino_fx # pylint: disable=unused-import
        torch._dynamo.eval_frame.check_if_dynamo_supported = lambda: True # pylint: disable=protected-access
        if shared.compiled_model_state is None:
            shared.compiled_model_state = CompiledModelState()
        else:
            if not shared.compiled_model_state.lora_compile:
                shared.compiled_model_state.lora_compile = False
                shared.compiled_model_state.lora_model = []
            shared.compiled_model_state.compiled_cache.clear()
            shared.compiled_model_state.partitioned_modules.clear()
        shared.compiled_model_state.first_pass = True if not shared.opts.cuda_compile_precompile else False
    except Exception as e:
        shared.log.warning(f"Model compile: task=OpenVINO: {e}")


def compile_stablefast(sd_model):
    try:
        import sfast.compilers.stable_diffusion_pipeline_compiler as sf
    except Exception as e:
        shared.log.warning(f'Model compile using stable-fast: {e}')
        return sd_model
    config = sf.CompilationConfig.Default()
    try:
        import xformers # pylint: disable=unused-import
        config.enable_xformers = True
    except Exception:
        pass
    try:
        import triton # pylint: disable=unused-import
        config.enable_triton = True
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    config.enable_cuda_graph = shared.opts.cuda_compile_fullgraph
    config.enable_jit_freeze = shared.opts.diffusers_eval
    config.memory_format = torch.channels_last if shared.opts.opt_channelslast else torch.contiguous_format
    # config.enable_cnn_optimization
    # config.prefer_lowp_gemm
    try:
        t0 = time.time()
        sd_model = sf.compile(sd_model, config)
        setup_logging() # compile messes with logging so reset is needed
        if shared.opts.cuda_compile_precompile:
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: task=Stable-fast config={config.__dict__} time={t1-t0:.2f}")
    except Exception as e:
        shared.log.info(f"Model compile: task=Stable-fast error: {e}")
    return sd_model


def compile_torch(sd_model):
    try:
        import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
        torch._dynamo.reset() # pylint: disable=protected-access
        shared.log.debug(f"Model compile available backends: {torch._dynamo.list_backends()}") # pylint: disable=protected-access
        if shared.opts.ipex_optimize:
            optimize_ipex(sd_model)
        if shared.opts.cuda_compile_backend == "openvino_fx":
            optimize_openvino()
        log_level = logging.WARNING if shared.opts.cuda_compile_verbose else logging.CRITICAL # pylint: disable=protected-access
        if hasattr(torch, '_logging'):
            torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
        torch._dynamo.config.verbose = shared.opts.cuda_compile_verbose # pylint: disable=protected-access
        torch._dynamo.config.suppress_errors = shared.opts.cuda_compile_errors # pylint: disable=protected-access
        t0 = time.time()
        if shared.opts.cuda_compile:
            sd_model.unet = torch.compile(sd_model.unet, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph)
        if shared.opts.cuda_compile_vae:
            if hasattr(sd_model, 'vae'):
                sd_model.vae.decode = torch.compile(sd_model.vae.decode, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph)
            if hasattr(sd_model, 'movq'):
                sd_model.movq.decode = torch.compile(sd_model.movq.decode, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph)
        setup_logging() # compile messes with logging so reset is needed
        if shared.opts.cuda_compile_precompile:
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Model compile error: {e}")
    return sd_model


def compile_diffusers(sd_model):
    if not (shared.opts.cuda_compile or shared.opts.cuda_compile_vae or shared.opts.cuda_compile_upscaler):
        return sd_model
    if not hasattr(sd_model, 'unet') or not hasattr(sd_model.unet, 'config'):
        shared.log.warning('Model compile enabled but model has no Unet')
        return sd_model
    if shared.opts.cuda_compile_backend == 'none':
        shared.log.warning('Model compile enabled but no backend specified')
        return sd_model
    size = 8*getattr(sd_model.unet.config, 'sample_size', 0)
    shared.log.info(f"Model compile: pipeline={sd_model.__class__.__name__} shape={size} mode={shared.opts.cuda_compile_mode} backend={shared.opts.cuda_compile_backend} fullgraph={shared.opts.cuda_compile_fullgraph} unet={shared.opts.cuda_compile} vae={shared.opts.cuda_compile_vae} upscaler={shared.opts.cuda_compile_upscaler}")
    if shared.opts.cuda_compile_backend == 'stable-fast':
        sd_model = compile_stablefast(sd_model)
    else:
        sd_model = compile_torch(sd_model)
    return sd_model
