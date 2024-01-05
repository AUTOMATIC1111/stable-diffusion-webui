import time
import logging
import torch
from modules import shared, devices
from installer import setup_logging


#Used by OpenVINO, can be used with TensorRT or Olive
class CompiledModelState:
    def __init__(self):
        self.model_str = ""
        self.first_pass = True
        self.first_pass_refiner = True
        self.first_pass_vae = True
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.partition_id = 0
        self.cn_model = []
        self.lora_model = []
        self.lora_compile = False
        self.compiled_cache = {}
        self.partitioned_modules = {}
        self.compiling_vae = False


def ipex_optimize(sd_model):
    try:
        t0 = time.time()
        import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
        if hasattr(sd_model, 'unet'):
            sd_model.unet.training = False
            sd_model.unet = ipex.optimize(sd_model.unet, dtype=devices.dtype_unet, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
        else:
            shared.log.warning('IPEX Optimize enabled but model has no Unet')
        if hasattr(sd_model, 'vae'):
            sd_model.vae.training = False
            sd_model.vae = ipex.optimize(sd_model.vae, dtype=devices.dtype_vae, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
        if hasattr(sd_model, 'movq'):
            sd_model.movq.training = False
            sd_model.movq = ipex.optimize(sd_model.movq, dtype=devices.dtype_vae, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
        t1 = time.time()
        shared.log.info(f"IPEX Optimize: time={t1-t0:.2f}")
        return sd_model
    except Exception as e:
        shared.log.warning(f"IPEX Optimize: error: {e}")

def nncf_compress_weights(sd_model):
    try:
        t0 = time.time()
        import nncf
        if hasattr(sd_model, 'unet'):
            sd_model.unet = nncf.compress_weights(sd_model.unet)
        else:
            shared.log.warning('Compress Weights enabled but model has no Unet')
        if shared.opts.nncf_compress_vae_weights:
            if hasattr(sd_model, 'vae'):
                sd_model.vae = nncf.compress_weights(sd_model.vae)
            if hasattr(sd_model, 'movq'):
                sd_model.movq = nncf.compress_weights(sd_model.movq)
        t1 = time.time()
        shared.log.info(f"Compress Weights: time={t1-t0:.2f}")
        return sd_model
    except Exception as e:
        shared.log.warning(f"Compress Weights: error: {e}")


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
            shared.compiled_model_state.partition_id = 0
            shared.compiled_model_state.model_str = ""
        shared.compiled_model_state.first_pass = True if not shared.opts.cuda_compile_precompile else False
        shared.compiled_model_state.first_pass_vae = True if not shared.opts.cuda_compile_precompile else False
        shared.compiled_model_state.first_pass_refiner = True if not shared.opts.cuda_compile_precompile else False
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
    # config.trace_scheduler = False
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
        if shared.opts.cuda_compile_backend == "openvino_fx":
            optimize_openvino()
        log_level = logging.WARNING if shared.opts.cuda_compile_verbose else logging.CRITICAL # pylint: disable=protected-access
        if hasattr(torch, '_logging'):
            torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
        torch._dynamo.config.verbose = shared.opts.cuda_compile_verbose # pylint: disable=protected-access
        torch._dynamo.config.suppress_errors = shared.opts.cuda_compile_errors # pylint: disable=protected-access

        try:
            torch._inductor.config.conv_1x1_as_mm = True # pylint: disable=protected-access
            torch._inductor.config.coordinate_descent_tuning = True # pylint: disable=protected-access
            torch._inductor.config.epilogue_fusion = False # pylint: disable=protected-access
            torch._inductor.config.coordinate_descent_check_all_directions = True # pylint: disable=protected-access
            torch._inductor.config.use_mixed_mm = True # pylint: disable=protected-access
            # torch._inductor.config.force_fuse_int_mm_with_mul = True # pylint: disable=protected-access
        except Exception as e:
            shared.log.error(f"Torch inductor config error: {e}")

        t0 = time.time()
        if shared.opts.cuda_compile:
            if shared.opts.cuda_compile and (not hasattr(sd_model, 'unet') or not hasattr(sd_model.unet, 'config')):
                shared.log.warning('Model compile enabled but model has no Unet')
            else:
                sd_model.unet = torch.compile(sd_model.unet, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph)
        if shared.opts.cuda_compile_vae:
            if hasattr(sd_model, 'vae') and hasattr(sd_model.vae, 'decode'):
                sd_model.vae.decode = torch.compile(sd_model.vae.decode, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph)
            elif hasattr(sd_model, 'movq') and hasattr(sd_model.movq, 'decode'):
                sd_model.movq.decode = torch.compile(sd_model.movq.decode, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph)
            else:
                shared.log.warning('Model compile enabled but model has no VAE')
        setup_logging() # compile messes with logging so reset is needed
        if shared.opts.cuda_compile_precompile:
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Model compile error: {e}")
    return sd_model


def compile_diffusers(sd_model):
    if shared.opts.ipex_optimize:
        sd_model = ipex_optimize(sd_model)
    if shared.opts.nncf_compress_weights and not (shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx"):
        sd_model = nncf_compress_weights(sd_model)
    if not (shared.opts.cuda_compile or shared.opts.cuda_compile_vae or shared.opts.cuda_compile_upscaler):
        return sd_model
    if shared.opts.cuda_compile_backend == 'none':
        shared.log.warning('Model compile enabled but no backend specified')
        return sd_model
    shared.log.info(f"Model compile: pipeline={sd_model.__class__.__name__} mode={shared.opts.cuda_compile_mode} backend={shared.opts.cuda_compile_backend} fullgraph={shared.opts.cuda_compile_fullgraph} unet={shared.opts.cuda_compile} vae={shared.opts.cuda_compile_vae} upscaler={shared.opts.cuda_compile_upscaler}")
    if shared.opts.cuda_compile_backend == 'stable-fast':
        sd_model = compile_stablefast(sd_model)
    else:
        sd_model = compile_torch(sd_model)
    return sd_model


def dynamic_quantization(sd_model):
    try:
        from torchao.quantization import quant_api
    except Exception as e:
        shared.log.error(f"Model dynamic quantization not supported: {e}")
        return sd_model

    def dynamic_quant_filter_fn(mod, *args): # pylint: disable=unused-argument
        return (isinstance(mod, torch.nn.Linear) and mod.in_features > 16 and (mod.in_features, mod.out_features)
                not in [(1280, 640), (1920, 1280), (1920, 640), (2048, 1280), (2048, 2560), (2560, 1280), (256, 128), (2816, 1280), (320, 640), (512, 1536), (512, 256), (512, 512), (640, 1280), (640, 1920), (640, 320), (640, 5120), (640, 640), (960, 320), (960, 640)])

    def conv_filter_fn(mod, *args): # pylint: disable=unused-argument
        return (isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels])

    shared.log.info(f"Model dynamic quantization: pipeline={sd_model.__class__.__name__}")
    try:
        quant_api.swap_conv2d_1x1_to_linear(sd_model.unet, conv_filter_fn)
        quant_api.swap_conv2d_1x1_to_linear(sd_model.vae, conv_filter_fn)
        quant_api.apply_dynamic_quant(sd_model.unet, dynamic_quant_filter_fn)
        quant_api.apply_dynamic_quant(sd_model.vae, dynamic_quant_filter_fn)
    except Exception as e:
        shared.log.error(f"Model dynamic quantization error: {e}")
    return sd_model
