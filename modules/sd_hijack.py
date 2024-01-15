from types import MethodType, SimpleNamespace
import io
import contextlib
import torch
from torch.nn.functional import silu

from modules import shared
shared.log.debug('Importing LDM')
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    import ldm.modules.attention
    import ldm.modules.distributions.distributions
    import ldm.modules.diffusionmodules.model
    import ldm.modules.diffusionmodules.openaimodel
    import ldm.models.diffusion.ddim
    import ldm.models.diffusion.plms
    import ldm.modules.encoders.modules

import modules.textual_inversion.textual_inversion
from modules import devices, sd_hijack_optimizations
from modules import sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr
from modules.hypernetworks import hypernetwork

attention_CrossAttention_forward = ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward

# new memory efficient cross attention blocks do not support hypernets and we already
# have memory efficient cross attention anyway, so this disables SD2.0's memory efficient cross attention
ldm.modules.attention.MemoryEfficientCrossAttention = ldm.modules.attention.CrossAttention
ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES["softmax-xformers"] = ldm.modules.attention.CrossAttention

# silence new console spam from SD2
ldm.modules.attention.print = lambda *args: None
ldm.modules.diffusionmodules.model.print = lambda *args: None

current_optimizer = SimpleNamespace(**{ "name": "none" })

def apply_optimizations():
    undo_optimizations()
    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th
    optimization_method = None
    can_use_sdp = hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(torch.nn.functional.scaled_dot_product_attention)
    if devices.device == torch.device("cpu"):
        if shared.opts.cross_attention_optimization == "Scaled-Dot-Product":
            shared.log.warning("Cross-attention: Scaled dot product is not available on CPU")
            can_use_sdp = False
        if shared.opts.cross_attention_optimization == "xFormers":
            shared.log.warning("Cross-attention: xFormers is not available on CPU")
            shared.xformers_available = False

    shared.log.info(f"Cross-attention: optimization={shared.opts.cross_attention_optimization} options={shared.opts.cross_attention_options}")
    if shared.opts.cross_attention_optimization == "Disabled":
        optimization_method = 'none'
    if can_use_sdp and shared.opts.cross_attention_optimization == "Scaled-Dot-Product" and 'SDP disable memory attention' in shared.opts.cross_attention_options:
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.scaled_dot_product_no_mem_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.sdp_no_mem_attnblock_forward
        optimization_method = 'sdp-no-mem'
    elif can_use_sdp and shared.opts.cross_attention_optimization == "Scaled-Dot-Product":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.scaled_dot_product_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.sdp_attnblock_forward
        optimization_method = 'sdp'
    if shared.xformers_available and shared.opts.cross_attention_optimization == "xFormers":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.xformers_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.xformers_attnblock_forward
        optimization_method = 'xformers'
    if shared.opts.cross_attention_optimization == "Sub-quadratic":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.sub_quad_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.sub_quad_attnblock_forward
        optimization_method = 'sub-quadratic'
    if shared.opts.cross_attention_optimization == "Split attention":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_v1
        optimization_method = 'v1'
    if shared.opts.cross_attention_optimization == "InvokeAI's":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_invokeAI
        optimization_method = 'invokeai'
    if shared.opts.cross_attention_optimization == "Doggettx's":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.cross_attention_attnblock_forward
        optimization_method = 'doggettx'
    current_optimizer.name = optimization_method
    return optimization_method


def undo_optimizations():
    ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def fix_checkpoint():
    """checkpoints are now added and removed in embedding/hypernet code, since torch doesn't want
    checkpoints to be added when not training (there's a warning)"""
    pass # pylint: disable=unnecessary-pass


def weighted_loss(sd_model, pred, target, mean=True):
    #Calculate the weight normally, but ignore the mean
    loss = sd_model._old_get_loss(pred, target, mean=False) # pylint: disable=protected-access

    #Check if we have weights available
    weight = getattr(sd_model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight

    #Return the loss, as mean if specified
    return loss.mean() if mean else loss

def weighted_forward(sd_model, x, c, w, *args, **kwargs):
    try:
        #Temporarily append weights to a place accessible during loss calc
        sd_model._custom_loss_weight = w # pylint: disable=protected-access

        #Replace 'get_loss' with a weight-aware one. Otherwise we need to reimplement 'forward' completely
        #Keep 'get_loss', but don't overwrite the previous old_get_loss if it's already set
        if not hasattr(sd_model, '_old_get_loss'):
            sd_model._old_get_loss = sd_model.get_loss # pylint: disable=protected-access
        sd_model.get_loss = MethodType(weighted_loss, sd_model)

        #Run the standard forward function, but with the patched 'get_loss'
        return sd_model.forward(x, c, *args, **kwargs)
    finally:
        try:
            #Delete temporary weights if appended
            del sd_model._custom_loss_weight
        except AttributeError:
            pass

        #If we have an old loss function, reset the loss function to the original one
        if hasattr(sd_model, '_old_get_loss'):
            sd_model.get_loss = sd_model._old_get_loss # pylint: disable=protected-access
            del sd_model._old_get_loss

def apply_weighted_forward(sd_model):
    #Add new function 'weighted_forward' that can be called to calc weighted loss
    sd_model.weighted_forward = MethodType(weighted_forward, sd_model)

def undo_weighted_forward(sd_model):
    try:
        del sd_model.weighted_forward
    except AttributeError:
        pass


class StableDiffusionModelHijack:
    fixes = None
    comments = []
    layers = None
    circular_enabled = False
    clip = None
    optimization_method = None

    embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()

    def __init__(self):
        self.embedding_db.add_embedding_dir(shared.opts.embeddings_dir)

    def hijack(self, m):
        if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation:
            model_embeddings = m.cond_stage_model.roberta.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.word_embeddings, self)
            m.cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            m.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
            m.cond_stage_model.model.token_embedding = EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding, self)
            m.cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        apply_weighted_forward(m)
        if m.cond_stage_key == "edit":
            sd_hijack_unet.hijack_ddpm_edit()

        if "Model" in shared.opts.ipex_optimize and shared.backend == shared.Backend.ORIGINAL:
            try:
                import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
                m.model.training = False
                m.model = ipex.optimize(m.model, dtype=devices.dtype_unet, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
                shared.log.info("Applied IPEX Optimize.")
            except Exception as err:
                shared.log.warning(f"IPEX Optimize not supported: {err}")

        if "Model" in shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none' and shared.backend == shared.Backend.ORIGINAL:
            try:
                import logging
                shared.log.info(f"Compiling pipeline={m.model.__class__.__name__} mode={shared.opts.cuda_compile_backend}")
                import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
                log_level = logging.WARNING if shared.opts.cuda_compile_verbose else logging.CRITICAL # pylint: disable=protected-access
                if hasattr(torch, '_logging'):
                    torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
                torch._dynamo.config.verbose = shared.opts.cuda_compile_verbose # pylint: disable=protected-access
                torch._dynamo.config.suppress_errors = shared.opts.cuda_compile_errors # pylint: disable=protected-access
                torch.backends.cudnn.benchmark = True
                if shared.opts.cuda_compile_backend == 'hidet':
                    import hidet # pylint: disable=import-error
                    hidet.torch.dynamo_config.use_tensor_core(True)
                    hidet.torch.dynamo_config.search_space(2)
                m.model = torch.compile(m.model, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph, dynamic=False)
                shared.log.info("Model complilation done.")
            except Exception as err:
                shared.log.warning(f"Model compile not supported: {err}")
            finally:
                from installer import setup_logging
                setup_logging()

        self.optimization_method = apply_optimizations()
        self.clip = m.cond_stage_model

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)

    def undo_hijack(self, m):
        if not hasattr(m, 'cond_stage_model'):
            return # not ldm model
        if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation:
            m.cond_stage_model = m.cond_stage_model.wrapped
        elif type(m.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        elif type(m.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
            m.cond_stage_model = m.cond_stage_model.wrapped
        undo_optimizations()
        undo_weighted_forward(m)
        self.apply_circular(False)
        self.layers = None
        self.clip = None

    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return
        self.circular_enabled = enable
        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def clear_comments(self):
        self.comments = []

    def get_prompt_lengths(self, text):
        if self.clip is None:
            return 0, 0
        _, token_count = self.clip.process_texts([text])
        return token_count, self.clip.get_target_prompt_token_count(token_count)


class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = devices.cond_cast_unet(embedding.vec)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            vecs.append(tensor)

        return torch.stack(vecs)


def add_circular_option_to_conv_2d():
    conv2d_constructor = torch.nn.Conv2d.__init__

    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode='circular', **kwargs)

    torch.nn.Conv2d.__init__ = conv2d_constructor_circular


model_hijack = StableDiffusionModelHijack()


def register_buffer(self, name, attr):
    """
    Fix register buffer bug for Mac OS.
    """

    if type(attr) == torch.Tensor:
        if attr.device != devices.device:
            attr = attr.to(device=devices.device, dtype=(torch.float32 if devices.device.type == 'mps' else None))

    setattr(self, name, attr)


ldm.models.diffusion.ddim.DDIMSampler.register_buffer = register_buffer
ldm.models.diffusion.plms.PLMSSampler.register_buffer = register_buffer

# Ensure samping from Guassian for DDPM follows types
ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample = lambda self: self.mean.to(self.parameters.dtype) + self.std.to(self.parameters.dtype) * torch.randn(self.mean.shape, dtype=self.parameters.dtype).to(device=self.parameters.device)
