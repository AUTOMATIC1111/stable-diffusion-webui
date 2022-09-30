import inspect
from modules.processing import Processed, process_images
import gradio as gr
import modules.scripts as scripts
import k_diffusion.sampling
import torch


class Script(scripts.Script):

    def title(self):
        return "Alternate Sampler Noise Schedules"

    def ui(self, is_img2img):
      noise_scheduler = gr.Dropdown(label="Noise Scheduler", choices=['Default','Karras','Exponential', 'Variance Preserving'], value='Default', type="index")
      sched_smin     = gr.Slider(value=0.1,   label="Sigma min",                  minimum=0.0,   maximum=100.0, step=0.5,)
      sched_smax     = gr.Slider(value=10.0,  label="Sigma max",                  minimum=0.0,   maximum=100.0, step=0.5)
      sched_rho      = gr.Slider(value=7.0,   label="Sigma rho (Karras only)",    minimum=7.0,   maximum=100.0, step=0.5)
      sched_beta_d   = gr.Slider(value=19.9,  label="Beta distribution (VP only)",minimum=0.0,   maximum=40.0, step=0.5)
      sched_beta_min = gr.Slider(value=0.1,   label="Beta min (VP only)",         minimum=0.0,   maximum=40.0, step=0.1)
      sched_eps_s    = gr.Slider(value=0.001, label="Epsilon (VP only)",          minimum=0.001, maximum=1.0,   step=0.001)

      return [noise_scheduler, sched_smin, sched_smax, sched_rho, sched_beta_d, sched_beta_min, sched_eps_s]

    def run(self, p, noise_scheduler, sched_smin, sched_smax, sched_rho, sched_beta_d, sched_beta_min, sched_eps_s):
      
      noise_scheduler_func_name = ['-','get_sigmas_karras','get_sigmas_exponential','get_sigmas_vp'][noise_scheduler]

      base_params = {
        "sigma_min":sched_smin, 
        "sigma_max":sched_smax, 
        "rho":sched_rho, 
        "beta_d":sched_beta_d, 
        "beta_min":sched_beta_min, 
        "eps_s":sched_eps_s,
        "device":"cuda" if torch.cuda.is_available() else "cpu"
      }

      if hasattr(k_diffusion.sampling,noise_scheduler_func_name):

        sigma_func  = getattr(k_diffusion.sampling,noise_scheduler_func_name)
        sigma_func_kwargs = {}

        for k,v in base_params.items():
          if k in inspect.signature(sigma_func).parameters:
            sigma_func_kwargs[k] = v

        def substitute_noise_scheduler(n):
          return sigma_func(n,**sigma_func_kwargs)

        p.sampler_noise_scheduler_override = substitute_noise_scheduler

      return process_images(p)