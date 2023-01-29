import torch
from torch.nn.functional import silu

import modules.textual_inversion.textual_inversion
from modules import devices, sd_hijack_optimizations, shared, sd_hijack_checkpoint
from modules.hypernetworks import hypernetwork
from modules.shared import cmd_opts
from modules import sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr

import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import ldm.modules.diffusionmodules.openaimodel
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
import ldm.modules.encoders.modules

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


def apply_optimizations():
    undo_optimizations()

    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th
    
    optimization_method = None

    if cmd_opts.force_enable_xformers or (cmd_opts.xformers and shared.xformers_available and torch.version.cuda and (6, 0) <= torch.cuda.get_device_capability(shared.device) <= (9, 0)):
        print("Applying xformers cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.xformers_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.xformers_attnblock_forward
        optimization_method = 'xformers'
    elif cmd_opts.opt_sub_quad_attention:
        print("Applying sub-quadratic cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.sub_quad_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.sub_quad_attnblock_forward
        optimization_method = 'sub-quadratic'
    elif cmd_opts.opt_split_attention_v1:
        print("Applying v1 cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_v1
        optimization_method = 'V1'
    elif not cmd_opts.disable_opt_split_attention and (cmd_opts.opt_split_attention_invokeai or not cmd_opts.opt_split_attention and not torch.cuda.is_available()):
        print("Applying cross attention optimization (InvokeAI).")
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_invokeAI
        optimization_method = 'InvokeAI'
    elif not cmd_opts.disable_opt_split_attention and (cmd_opts.opt_split_attention or torch.cuda.is_available()):
        print("Applying cross attention optimization (Doggettx).")
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.cross_attention_attnblock_forward
        optimization_method = 'Doggettx'

    return optimization_method


def undo_optimizations():
    ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def fix_checkpoint():
    """checkpoints are now added and removed in embedding/hypernet code, since torch doesn't want
    checkpoints to be added when not training (there's a warning)"""

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
        self.embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)

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
