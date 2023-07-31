import diffusers
from modules import shared

lora_state = { # TODO Lora state for Diffusers
    'multiplier': 1.0,
    'active': False,
    'loaded': 0,
}

def unload_diffusers_lora():
    try:
        pipe = shared.sd_model
        pipe.unload_lora_weights()
        lora_state['active'] = False
        lora_state['loaded'] = 0
        pipe._remove_text_encoder_monkey_patch() # pylint: disable=W0212
        proc_cls_name = next(iter(pipe.unet.attn_processors.values())).__class__.__name__
        non_lora_proc_cls = getattr(diffusers.models.attention_processor, proc_cls_name[len("LORA"):])
        pipe.unet.set_attn_processor(non_lora_proc_cls())
        # shared.log.debug('Diffusers LoRA unloaded')
    except Exception:
        pass


def load_diffusers_lora(name, lora, strength = 1.0):
    try:
        pipe = shared.sd_model
        pipe.load_lora_weights(lora.filename, cache_dir=shared.opts.diffusers_dir, local_files_only=True, lora_scale=strength)
        lora_state['active'] = True
        lora_state['loaded'] += 1
        lora_state['multiplier'] = strength
        # pipe.unet.load_attn_procs("pcuenq/pokemon-lora")
        shared.log.info(f"Diffusers LoRA loaded: {name} {lora_state['multiplier']}")
    except Exception as e:
        shared.log.error(f"Diffusers LoRA loading failed: {name} {e}")
