import datetime
import glob
import html
import os
import sys
import traceback
import tqdm

import torch

from ldm.util import default
from modules import devices, shared, processing, sd_models
import torch
from torch import einsum
from einops import rearrange, repeat
import modules.textual_inversion.dataset


class HypernetworkModule(torch.nn.Module):
    def __init__(self, dim, state_dict=None):
        super().__init__()

        self.linear1 = torch.nn.Linear(dim, dim * 2)
        self.linear2 = torch.nn.Linear(dim * 2, dim)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)
        else:
            self.linear1.weight.data.fill_(0.0001)
            self.linear1.bias.data.fill_(0.0001)
            self.linear2.weight.data.fill_(0.0001)
            self.linear2.bias.data.fill_(0.0001)

        self.to(devices.device)

    def forward(self, x):
        return x + (self.linear2(self.linear1(x)))


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, name=None):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

        for size in [320, 640, 768, 1280]:
            self.layers[size] = (HypernetworkModule(size), HypernetworkModule(size))

    def weights(self):
        res = []

        for k, layers in self.layers.items():
            for layer in layers:
                layer.train()
                res += [layer.linear1.weight, layer.linear1.bias, layer.linear2.weight, layer.linear2.bias]

        return res

    def save(self, filename):
        state_dict = {}

        for k, v in self.layers.items():
            state_dict[k] = (v[0].state_dict(), v[1].state_dict())

        state_dict['step'] = self.step
        state_dict['name'] = self.name
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name

        torch.save(state_dict, filename)

    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cpu')

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (HypernetworkModule(size, sd[0]), HypernetworkModule(size, sd[1]))

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)


def load_hypernetworks(path):
    res = {}

    for filename in glob.iglob(path + '**/*.pt', recursive=True):
        try:
            hn = Hypernetwork()
            hn.load(filename)
            res[hn.name] = hn
        except Exception:
            print(f"Error loading hypernetwork {filename}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    return res


def attention_CrossAttention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    hypernetwork_layers = (shared.hypernetwork.layers if shared.hypernetwork is not None else {}).get(context.shape[2], None)

    if hypernetwork_layers is not None:
        hypernetwork_k, hypernetwork_v = hypernetwork_layers

        self.hypernetwork_k = hypernetwork_k
        self.hypernetwork_v = hypernetwork_v

        context_k = hypernetwork_k(context)
        context_v = hypernetwork_v(context)
    else:
        context_k = context
        context_v = context

    k = self.to_k(context_k)
    v = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


def train_hypernetwork(hypernetwork_name, learn_rate, data_root, log_directory, steps, create_image_every, save_hypernetwork_every, template_file, preview_image_prompt):
    assert hypernetwork_name, 'embedding not selected'

    shared.hypernetwork = shared.hypernetworks[hypernetwork_name]

    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)

    if save_hypernetwork_every > 0:
        hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
        os.makedirs(hypernetwork_dir, exist_ok=True)
    else:
        hypernetwork_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    cond_model = shared.sd_model.cond_stage_model

    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, size=512, placeholder_token=hypernetwork_name, model=shared.sd_model, device=devices.device, template_file=template_file)

    hypernetwork = shared.hypernetworks[hypernetwork_name]
    weights = hypernetwork.weights()
    for weight in weights:
        weight.requires_grad = True

    optimizer = torch.optim.AdamW(weights, lr=learn_rate)

    losses = torch.zeros((32,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"

    ititial_step = hypernetwork.step or 0
    if ititial_step > steps:
        return hypernetwork, filename

    pbar = tqdm.tqdm(enumerate(ds), total=steps-ititial_step)
    for i, (x, text) in pbar:
        hypernetwork.step = i + ititial_step

        if hypernetwork.step > steps:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = cond_model([text])

            x = x.to(devices.device)
            loss = shared.sd_model(x.unsqueeze(0), c)[0]
            del x

            losses[hypernetwork.step % losses.shape[0]] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description(f"loss: {losses.mean():.7f}")

        if hypernetwork.step > 0 and hypernetwork_dir is not None and hypernetwork.step % save_hypernetwork_every == 0:
            last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork_name}-{hypernetwork.step}.pt')
            hypernetwork.save(last_saved_file)

        if hypernetwork.step > 0 and images_dir is not None and hypernetwork.step % create_image_every == 0:
            last_saved_image = os.path.join(images_dir, f'{hypernetwork_name}-{hypernetwork.step}.png')

            preview_text = text if preview_image_prompt == "" else preview_image_prompt

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=preview_text,
                steps=20,
                do_not_save_grid=True,
                do_not_save_samples=True,
            )

            processed = processing.process_images(p)
            image = processed.images[0]

            shared.state.current_image = image
            image.save(last_saved_image)

            last_saved_image += f", prompt: {preview_text}"

        shared.state.job_no = hypernetwork.step

        shared.state.textinfo = f"""
<p>
Loss: {losses.mean():.7f}<br/>
Step: {hypernetwork.step}<br/>
Last prompt: {html.escape(text)}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    checkpoint = sd_models.select_checkpoint()

    hypernetwork.sd_checkpoint = checkpoint.hash
    hypernetwork.sd_checkpoint_name = checkpoint.model_name
    hypernetwork.save(filename)

    return hypernetwork, filename


