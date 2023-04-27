import argparse
import os
from modules.paths_internal import data_path, sd_default_config, sd_model_file

parser = argparse.ArgumentParser(description="Stable Diffusion", formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=55,indent_increment=2,width=200))

parser.add_argument("-f", action='store_true', help=argparse.SUPPRESS)  # allows running as root; implemented outside of webui
parser.add_argument("--ui-settings-file", type=str, help=argparse.SUPPRESS, default=os.path.join(data_path, 'config.json'))
parser.add_argument("--ui-config-file", type=str, help=argparse.SUPPRESS, default=os.path.join(data_path, 'ui-config.json'))
parser.add_argument("--config", type=str, default=sd_default_config, help=argparse.SUPPRESS)
parser.add_argument("--theme", type=str, help=argparse.SUPPRESS, default=None)

parser.add_argument("--medvram", action='store_true', help="Enable model optimizations for sacrificing a little speed for low memory usage")
parser.add_argument("--lowvram", action='store_true', help="Enable model optimizations for sacrificing a lot of speed for lowest memory usage")
parser.add_argument("--lowram", action='store_true', help="Load checkpoint weights to VRAM instead of RAM")

parser.add_argument("--ckpt", type=str, default=sd_model_file, help="Path to checkpoint of stable diffusion model to load immediately",)
parser.add_argument('--vae', type=str, help='Path to checkpoint of stable diffusion VAE model to load immediately', default=None)
parser.add_argument("--data-dir", type=str, default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))), help="Base path where all user data is stored")
parser.add_argument("--models-dir", type=str, default="models", help="Nase path where all models are stored",)

parser.add_argument("--allow-code", action='store_true', help="Allow custom script execution")
parser.add_argument("--share", action='store_true', help="Enable to make the UI accessible through Gradio site")
parser.add_argument("--enable-insecure", action='store_true', help="Enable extensions tab regardless of other options")
parser.add_argument("--use-cpu", nargs='+', help="Force use CPU for specified modules", default=[], type=str.lower)
parser.add_argument("--listen", action='store_true', help="Launch web server using public IP address")
parser.add_argument("--port", type=int, help="Launch web server with given server port", default=None)
parser.add_argument("--hide-ui-dir-config", action='store_true', help="Hide directory configuration from UI", default=False)
parser.add_argument("--freeze-settings", action='store_true', help="Disable editing settings", default=False)
parser.add_argument("--gradio-auth", type=str, help='Set Gradio authentication like "username:password,username:password""', default=None)
parser.add_argument("--gradio-auth-path", type=str, help='Set Gradio authentication using file', default=None)
parser.add_argument("--autolaunch", action='store_true', help="Open the UI URL in the system's default browser upon launch", default=False)
parser.add_argument("--disable-console-progressbars", action='store_true', help="Do not output progressbars to console", default=True)
parser.add_argument("--disable-safe-unpickle", action='store_true', help="Disable checking models for malicious code", default=True)
parser.add_argument("--api-auth", type=str, help='Set API authentication', default=None)
parser.add_argument("--api-log", action='store_true', help="Enable logging of all API requests")
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use", default=None)
parser.add_argument("--cors-origins", type=str, help="Allowed CORS origin(s) in the form of a comma-separated list", default=None)
parser.add_argument("--cors-regex", type=str, help="Allowed CORS origin(s) in the form of a single regular expression", default=None)
parser.add_argument("--tls-keyfile", type=str, help="Partially enables TLS, requires --tls-certfile to fully function", default=None)
parser.add_argument("--tls-certfile", type=str, help="Partially enables TLS, requires --tls-keyfile to fully function", default=None)
parser.add_argument("--server-name", type=str, help="Sets hostname of server", default=None)
parser.add_argument("--no-hashing", action='store_true', help="Disable sha256 hashing of checkpoints", default=False)
parser.add_argument("--no-download-sd-model", action='store_true', help="Disable download of default model even if no model is found", default=False)
parser.add_argument("--profile", action='store_true', help="Run profiler, default: %(default)s")
parser.add_argument("--disable-queue", action='store_true', help="Disable Gradio queues and force use of HTTP instead of WebSockets, default: %(default)s")


def compatibility_args(opts, args):
    parser.add_argument("--ckpt-dir", type=str, help=argparse.SUPPRESS, default=opts.ckpt_dir)
    parser.add_argument("--vae-dir", type=str, help=argparse.SUPPRESS, default=opts.vae_dir)
    parser.add_argument("--embeddings-dir", type=str, help=argparse.SUPPRESS, default=opts.embeddings_dir)
    parser.add_argument("--embeddings-templates-dir", type=str, help=argparse.SUPPRESS, default=opts.embeddings_templates_dir)
    parser.add_argument("--hypernetwork-dir", type=str, help=argparse.SUPPRESS, default=opts.hypernetwork_dir)
    parser.add_argument("--codeformer-models-path", type=str, help=argparse.SUPPRESS, default=opts.codeformer_models_path)
    parser.add_argument("--gfpgan-models-path", type=str, help=argparse.SUPPRESS, default=opts.gfpgan_models_path)
    parser.add_argument("--esrgan-models-path", type=str, help=argparse.SUPPRESS, default=opts.esrgan_models_path)
    parser.add_argument("--bsrgan-models-path", type=str, help=argparse.SUPPRESS, default=opts.bsrgan_models_path)
    parser.add_argument("--realesrgan-models-path", type=str, help=argparse.SUPPRESS, default=opts.realesrgan_models_path)
    parser.add_argument("--scunet-models-path", help=argparse.SUPPRESS, default=opts.scunet_models_path)
    parser.add_argument("--swinir-models-path", help=argparse.SUPPRESS, default=opts.swinir_models_path)
    parser.add_argument("--ldsr-models-path", help=argparse.SUPPRESS, default=opts.ldsr_models_path)
    parser.add_argument("--clip-models-path", type=str, help=argparse.SUPPRESS, default=opts.clip_models_path)
    parser.add_argument("--disable-extension-access", default = False, action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--opt-channelslast", help=argparse.SUPPRESS, default=opts.opt_channelslast)
    parser.add_argument("--xformers", default = (opts.cross_attention_optimization == "xFormers"), action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--disable-nan-check", help=argparse.SUPPRESS, default=opts.disable_nan_check)
    parser.add_argument("--token-merging", help=argparse.SUPPRESS, default=opts.token_merging)
    parser.add_argument("--rollback-vae", help=argparse.SUPPRESS, default=opts.rollback_vae)
    parser.add_argument("--no-half", help=argparse.SUPPRESS, default=opts.no_half)
    parser.add_argument("--no-half-vae", help=argparse.SUPPRESS, default=opts.no_half_vae)
    parser.add_argument("--precision", help=argparse.SUPPRESS, default=opts.precision)
    parser.add_argument("--api", help=argparse.SUPPRESS, default=True)
    parser.add_argument("--sub-quad-q-chunk-size", help=argparse.SUPPRESS, default=opts.sub_quad_q_chunk_size)
    parser.add_argument("--sub-quad-kv-chunk-size", help=argparse.SUPPRESS, default=opts.sub_quad_kv_chunk_size)
    parser.add_argument("--sub-quad-chunk-threshold", help=argparse.SUPPRESS, default=opts.sub_quad_chunk_threshold)

    opts.use_old_emphasis_implementation = False
    opts.use_old_karras_scheduler_sigmas = False
    opts.no_dpmpp_sde_batch_determinism = False
    opts.use_old_hires_fix_width_height = False

    parser.add_argument("--lora-dir", help=argparse.SUPPRESS, default=opts.lora_dir)
    args = parser.parse_args()
    if 'lyco_dir' in args:
        args.lyco_dir = opts.lyco_dir
    return args
