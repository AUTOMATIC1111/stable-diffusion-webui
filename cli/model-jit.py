#!/usr/bin/env python
import os
import time
import functools
import argparse
import logging
import warnings
from dataclasses import dataclass

logging.getLogger("DeepSpeed").disabled = True
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

import torch
import diffusers

n_warmup = 5
n_traces = 10
n_runs = 100
args = {}
pipe = None
log = logging.getLogger("sd")


def setup_logging():
    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.traceback import install
    log.setLevel(logging.DEBUG)
    console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({ "traceback.border": "black", "traceback.border.syntax_error": "black", "inspect.value.border": "black" }))
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s', handlers=[logging.NullHandler()]) # redirect default logger to null
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=logging.DEBUG, console=console)
    rh.setLevel(logging.DEBUG)
    log.addHandler(rh)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)
    install(console=console, extra_lines=1, max_frames=10, width=console.width, word_wrap=False, indent_guides=False, suppress=[])


def generate_inputs():
    if args.type == 'sd15':
        sample = torch.randn(2, 4, 64, 64).half().cuda()
        timestep = torch.rand(1).half().cuda() * 999
        encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
        return sample, timestep, encoder_hidden_states
    if args.type == 'sdxl':
        sample = torch.randn(2, 4, 64, 64).half().cuda()
        timestep = torch.rand(1).half().cuda() * 999
        encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
        text_embeds = torch.randn(1, 77, 2048).half().cuda()
        return sample, timestep, encoder_hidden_states, text_embeds


def load_model():
    log.info(f'versions: torch={torch.__version__} diffusers={diffusers.__version__}')
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16,
        "safety_checker": None,
        "requires_safety_checker": False,
        "load_safety_checker": False,
        "load_connected_pipeline": True,
        "use_safetensors": True,
    }
    pipeline = diffusers.StableDiffusionPipeline if args.type == 'sd15' else diffusers.StableDiffusionXLPipeline
    global pipe # pylint: disable=global-statement
    t0 = time.time()
    pipe = pipeline.from_single_file(args.model, **diffusers_load_config).to('cuda')
    size = os.path.getsize(args.model)
    log.info(f'load: model={args.model} type={args.type} time={time.time() - t0:.3f}s size={size / 1024 / 1024:.3f}mb')


def load_trace(fn: str):

    @dataclass
    class UNet2DConditionOutput:
        sample: torch.FloatTensor

    class TracedUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = pipe.unet.in_channels
            self.device = pipe.unet.device

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)

    t0 = time.time()
    unet_traced = torch.jit.load(fn)
    pipe.unet = TracedUNet()
    size = os.path.getsize(fn)
    log.info(f'load: optimized={fn} time={time.time() - t0:.3f}s size={size / 1024 / 1024:.3f}mb')


def trace_model():
    log.info(f'tracing model: {args.model}')
    torch.set_grad_enabled(False)
    unet = pipe.unet
    unet.eval()
    # unet.to(memory_format=torch.channels_last)  # use channels_last memory format
    unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

    # warmup
    t0 = time.time()
    for _ in range(n_warmup):
        with torch.inference_mode():
            inputs = generate_inputs()
            _output = unet(*inputs)
    log.info(f'warmup: time={time.time() - t0:.3f}s passes={n_warmup}')

    # trace
    t0 = time.time()
    unet_traced = torch.jit.trace(unet, inputs, check_trace=True)
    unet_traced.eval()
    log.info(f'trace: time={time.time() - t0:.3f}s')

    # optimize graph
    t0 = time.time()
    for _ in range(n_traces):
        with torch.inference_mode():
            inputs = generate_inputs()
            _output = unet_traced(*inputs)
    log.info(f'optimize: time={time.time() - t0:.3f}s passes={n_traces}')

    # save the model
    if args.save:
        t0 = time.time()
        basename, _ext = os.path.splitext(args.model)
        fn = f"{basename}.pt"
        unet_traced.save(fn)
        size = os.path.getsize(fn)
        log.info(f'save: optimized={fn} time={time.time() - t0:.3f}s size={size / 1024 / 1024:.3f}mb')
        return fn

    pipe.unet = unet_traced
    return None


def benchmark_model(msg: str):
    with torch.inference_mode():
        inputs = generate_inputs()
        torch.cuda.synchronize()
        for n in range(n_runs):
            if n > n_runs / 10:
                t0 = time.time()
            _output = pipe.unet(*inputs)
        torch.cuda.synchronize()
        t1 = time.time()
        log.info(f"benchmark unet: {t1 - t0:.3f}s passes={n_runs} type={msg}")
        return t1 - t0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'SD.Next')
    parser.add_argument('--model', type=str, default='', required=True, help='model path')
    parser.add_argument('--type', type=str, default='sd15', choices=['sd15', 'sdxl'], required=False, help='model type, default: %(default)s')
    parser.add_argument('--benchmark', default = False, action='store_true', help = "run benchmarks, default: %(default)s")
    parser.add_argument('--trace', default = True, action='store_true', help = "run jit tracing, default: %(default)s")
    parser.add_argument('--save', default = False, action='store_true', help = "save optimized unet, default: %(default)s")
    args = parser.parse_args()
    setup_logging()
    log.info('sdnext model jit tracing')
    if not os.path.isfile(args.model):
        log.error(f"invalid model path: {args.model}")
        exit(1)
    load_model()
    if args.benchmark:
        time0 = benchmark_model('original')
    unet_saved = trace_model()
    if unet_saved is not None:
        load_trace(unet_saved)
    if args.benchmark:
        time1 = benchmark_model('traced')
        log.info(f'benchmark speedup: {100 * (time0 - time1) / time0:.3f}%')
