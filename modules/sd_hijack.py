import math
import os
import sys
import traceback
import torch
import numpy as np
from torch import einsum
from torch.nn.functional import silu

import modules.textual_inversion.textual_inversion
from modules import prompt_parser, devices, sd_hijack_optimizations, shared
from modules.shared import opts, device, cmd_opts
from modules.sd_hijack_optimizations import invokeAI_mps_available

import ldm.modules.attention
import ldm.modules.diffusionmodules.model

attention_CrossAttention_forward = ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward


def apply_optimizations():
    undo_optimizations()

    ldm.modules.diffusionmodules.model.nonlinearity = silu

    if cmd_opts.force_enable_xformers or (cmd_opts.xformers and shared.xformers_available and torch.version.cuda and (6, 0) <= torch.cuda.get_device_capability(shared.device) <= (9, 0)):
        print("Applying xformers cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.xformers_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.xformers_attnblock_forward
    elif cmd_opts.opt_split_attention_v1:
        print("Applying v1 cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_v1
    elif not cmd_opts.disable_opt_split_attention and (cmd_opts.opt_split_attention_invokeai or not torch.cuda.is_available()):
        if not invokeAI_mps_available and shared.device.type == 'mps':
            print("The InvokeAI cross attention optimization for MPS requires the psutil package which is not installed.")
            print("Applying v1 cross attention optimization.")
            ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_v1
        else:
            print("Applying cross attention optimization (InvokeAI).")
            ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_invokeAI
    elif not cmd_opts.disable_opt_split_attention and (cmd_opts.opt_split_attention or torch.cuda.is_available()):
        print("Applying cross attention optimization (Doggettx).")
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.cross_attention_attnblock_forward


def undo_optimizations():
    from modules.hypernetworks import hypernetwork

    ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def get_target_prompt_token_count(token_count):
    return math.ceil(max(token_count, 1) / 75) * 75


class StableDiffusionModelHijack:
    fixes = None
    comments = []
    layers = None
    circular_enabled = False
    clip = None

    embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase(cmd_opts.embeddings_dir)

    def hijack(self, m):
        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings

        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        self.clip = m.cond_stage_model

        apply_optimizations()

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)

    def undo_hijack(self, m):
        if type(m.cond_stage_model) == FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped

        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
        if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
            model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped

        self.layers = None
        self.circular_enabled = False
        self.clip = None

    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return

        self.circular_enabled = enable

        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def clear_comments(self):
        self.comments = []

    def tokenize(self, text):
        _, remade_batch_tokens, _, _, _, token_count = self.clip.process_text([text])
        return remade_batch_tokens[0], token_count, get_target_prompt_token_count(token_count)


class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, wrapped, hijack):
        super().__init__()
        self.wrapped = wrapped
        self.hijack: StableDiffusionModelHijack = hijack
        self.tokenizer = wrapped.tokenizer
        self.token_mults = {}

        self.comma_token = [v for k, v in self.tokenizer.get_vocab().items() if k == ',</w>'][0]

        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def tokenize_line(self, line, used_custom_terms, hijack_comments):
        id_end = self.wrapped.tokenizer.eos_token_id

        if opts.enable_emphasis:
            parsed = prompt_parser.parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.wrapped.tokenizer([text for text, _ in parsed], truncation=False, add_special_tokens=False)["input_ids"]

        fixes = []
        remade_tokens = []
        multipliers = []
        last_comma = -1

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, i)

                if token == self.comma_token:
                    last_comma = len(remade_tokens)
                elif opts.comma_padding_backtrack != 0 and max(len(remade_tokens), 1) % 75 == 0 and last_comma != -1 and len(remade_tokens) - last_comma <= opts.comma_padding_backtrack:
                    last_comma += 1
                    reloc_tokens = remade_tokens[last_comma:]
                    reloc_mults = multipliers[last_comma:]

                    remade_tokens = remade_tokens[:last_comma]
                    length = len(remade_tokens)

                    rem = int(math.ceil(length / 75)) * 75 - length
                    remade_tokens += [id_end] * rem + reloc_tokens
                    multipliers = multipliers[:last_comma] + [1.0] * rem + reloc_mults

                if embedding is None:
                    remade_tokens.append(token)
                    multipliers.append(weight)
                    i += 1
                else:
                    emb_len = int(embedding.vec.shape[0])
                    iteration = len(remade_tokens) // 75
                    if (len(remade_tokens) + emb_len) // 75 != iteration:
                        rem = (75 * (iteration + 1) - len(remade_tokens))
                        remade_tokens += [id_end] * rem
                        multipliers += [1.0] * rem
                        iteration += 1
                    fixes.append((iteration, (len(remade_tokens) % 75, embedding)))
                    remade_tokens += [0] * emb_len
                    multipliers += [weight] * emb_len
                    used_custom_terms.append((embedding.name, embedding.checksum()))
                    i += embedding_length_in_tokens

        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)

        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add

        return remade_tokens, fixes, multipliers, token_count

    def process_text(self, texts):
        used_custom_terms = []
        remade_batch_tokens = []
        hijack_comments = []
        hijack_fixes = []
        token_count = 0

        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, fixes, multipliers = cache[line]
            else:
                remade_tokens, fixes, multipliers, current_token_count = self.tokenize_line(line, used_custom_terms, hijack_comments)
                token_count = max(current_token_count, token_count)

                cache[line] = (remade_tokens, fixes, multipliers)

            remade_batch_tokens.append(remade_tokens)
            hijack_fixes.append(fixes)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    def process_text_old(self, text):
        id_start = self.wrapped.tokenizer.bos_token_id
        id_end = self.wrapped.tokenizer.eos_token_id
        maxlen = self.wrapped.max_length  # you get to stay at 77
        used_custom_terms = []
        remade_batch_tokens = []
        overflowing_words = []
        hijack_comments = []
        hijack_fixes = []
        token_count = 0

        cache = {}
        batch_tokens = self.wrapped.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        batch_multipliers = []
        for tokens in batch_tokens:
            tuple_tokens = tuple(tokens)

            if tuple_tokens in cache:
                remade_tokens, fixes, multipliers = cache[tuple_tokens]
            else:
                fixes = []
                remade_tokens = []
                multipliers = []
                mult = 1.0

                i = 0
                while i < len(tokens):
                    token = tokens[i]

                    embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, i)

                    mult_change = self.token_mults.get(token) if opts.enable_emphasis else None
                    if mult_change is not None:
                        mult *= mult_change
                        i += 1
                    elif embedding is None:
                        remade_tokens.append(token)
                        multipliers.append(mult)
                        i += 1
                    else:
                        emb_len = int(embedding.vec.shape[0])
                        fixes.append((len(remade_tokens), embedding))
                        remade_tokens += [0] * emb_len
                        multipliers += [mult] * emb_len
                        used_custom_terms.append((embedding.name, embedding.checksum()))
                        i += embedding_length_in_tokens

                if len(remade_tokens) > maxlen - 2:
                    vocab = {v: k for k, v in self.wrapped.tokenizer.get_vocab().items()}
                    ovf = remade_tokens[maxlen - 2:]
                    overflowing_words = [vocab.get(int(x), "") for x in ovf]
                    overflowing_text = self.wrapped.tokenizer.convert_tokens_to_string(''.join(overflowing_words))
                    hijack_comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

                token_count = len(remade_tokens)
                remade_tokens = remade_tokens + [id_end] * (maxlen - 2 - len(remade_tokens))
                remade_tokens = [id_start] + remade_tokens[0:maxlen - 2] + [id_end]
                cache[tuple_tokens] = (remade_tokens, fixes, multipliers)

            multipliers = multipliers + [1.0] * (maxlen - 2 - len(multipliers))
            multipliers = [1.0] + multipliers[0:maxlen - 2] + [1.0]

            remade_batch_tokens.append(remade_tokens)
            hijack_fixes.append(fixes)
            batch_multipliers.append(multipliers)
        return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    def forward(self, text):
        use_old = opts.use_old_emphasis_implementation
        if use_old:
            batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.process_text_old(text)
        else:
            batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.process_text(text)

        self.hijack.comments += hijack_comments

        if len(used_custom_terms) > 0:
            self.hijack.comments.append("Used embeddings: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

        if use_old:
            self.hijack.fixes = hijack_fixes
            return self.process_tokens(remade_batch_tokens, batch_multipliers)

        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            self.hijack.fixes = []
            for unfiltered in hijack_fixes:
                fixes = []
                for fix in unfiltered:
                    if fix[0] == i:
                        fixes.append(fix[1])
                self.hijack.fixes.append(fixes)

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                    multipliers.append(batch_multipliers[j][:75])
                else:
                    tokens.append([self.wrapped.tokenizer.eos_token_id] * 75)
                    multipliers.append([1.0] * 75)

            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), axis=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1

        return z

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        if not opts.use_old_emphasis_implementation:
            remade_batch_tokens = [[self.wrapped.tokenizer.bos_token_id] + x[:75] + [self.wrapped.tokenizer.eos_token_id] for x in remade_batch_tokens]
            batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]

        tokens = torch.asarray(remade_batch_tokens).to(device)
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)

        if opts.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(batch_multipliers_of_same_length).to(device)
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z


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
                emb = embedding.vec
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
