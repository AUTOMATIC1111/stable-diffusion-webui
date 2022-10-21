import datetime
import glob
import html
import os
import sys
import traceback
import tqdm
import csv

import torch

from ldm.util import default
from modules import devices, shared, processing, sd_models
import torch
from torch import einsum
from einops import rearrange, repeat
import modules.textual_inversion.dataset
from modules.textual_inversion import textual_inversion
from modules.textual_inversion.learn_schedule import LearnRateScheduler


class HypernetworkModule(torch.nn.Module):
    multiplier = 1.0

    def __init__(self, dim, state_dict=None, layer_structure=None, add_layer_norm=False, activation_func=None):
        super().__init__()

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            if activation_func == "relu":
                linears.append(torch.nn.ReLU())
            elif activation_func == "leakyrelu":
                linears.append(torch.nn.LeakyReLU())
            elif activation_func == 'linear' or activation_func is None:
                pass
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    layer.weight.data.normal_(mean=0.0, std=0.01)
                    layer.bias.data.zero_()

        self.to(devices.device)

    def fix_old_state_dict(self, state_dict):
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x):
        return x + self.linear(x) * self.multiplier

    def trainables(self):
        layer_structure = []
        for layer in self.linear:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                layer_structure += [layer.weight, layer.bias]
        return layer_structure


def apply_strength(value=None):
    HypernetworkModule.multiplier = value if value is not None else shared.opts.sd_hypernetwork_strength


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, name=None, enable_sizes=None, layer_structure=None, add_layer_norm=False, activation_func=None):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.layer_structure = layer_structure
        self.add_layer_norm = add_layer_norm
        self.activation_func = activation_func

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.add_layer_norm, self.activation_func),
                HypernetworkModule(size, None, self.layer_structure, self.add_layer_norm, self.activation_func),
            )

    def weights(self):
        res = []

        for k, layers in self.layers.items():
            for layer in layers:
                layer.train()
                res += layer.trainables()

        return res

    def save(self, filename):
        state_dict = {}

        for k, v in self.layers.items():
            state_dict[k] = (v[0].state_dict(), v[1].state_dict())

        state_dict['step'] = self.step
        state_dict['name'] = self.name
        state_dict['layer_structure'] = self.layer_structure
        state_dict['is_layer_norm'] = self.add_layer_norm
        state_dict['activation_func'] = self.activation_func
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name

        torch.save(state_dict, filename)

    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cpu')

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        self.activation_func = state_dict.get('activation_func', None)

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    HypernetworkModule(size, sd[0], self.layer_structure, self.add_layer_norm, self.activation_func),
                    HypernetworkModule(size, sd[1], self.layer_structure, self.add_layer_norm, self.activation_func),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)


def list_hypernetworks(path):
    res = {}
    for filename in glob.iglob(os.path.join(path, '**/*.pt'), recursive=True):
        name = os.path.splitext(os.path.basename(filename))[0]
        res[name] = filename
    return res


def load_hypernetwork(filename):
    path = shared.hypernetworks.get(filename, None)
    if path is not None:
        print(f"Loading hypernetwork {filename}")
        try:
            shared.loaded_hypernetwork = Hypernetwork()
            shared.loaded_hypernetwork.load(path)

        except Exception:
            print(f"Error loading hypernetwork {path}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    else:
        if shared.loaded_hypernetwork is not None:
            print(f"Unloading hypernetwork")

        shared.loaded_hypernetwork = None


def find_closest_hypernetwork_name(search: str):
    if not search:
        return None
    search = search.lower()
    applicable = [name for name in shared.hypernetworks if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return applicable[0]


def apply_hypernetwork(hypernetwork, context, layer=None):
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)

    if hypernetwork_layers is None:
        return context, context

    if layer is not None:
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    context_k = hypernetwork_layers[0](context)
    context_v = hypernetwork_layers[1](context)
    return context_k, context_v


def attention_CrossAttention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = apply_hypernetwork(shared.loaded_hypernetwork, context, self)
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


def stack_conds(conds):
    if len(conds) == 1:
        return torch.stack(conds)

    # same as in reconstruct_multicond_batch
    token_count = max([x.shape[0] for x in conds])
    for i in range(len(conds)):
        if conds[i].shape[0] != token_count:
            last_vector = conds[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - conds[i].shape[0], 1])
            conds[i] = torch.vstack([conds[i], last_vector_repeated])

    return torch.stack(conds)


def train_hypernetwork(hypernetwork_name, learn_rate, batch_size, data_root, log_directory, training_width, training_height, steps, create_image_every, save_hypernetwork_every, template_file, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    # images allows training previews to have infotext. Importing it at the top causes a circular import problem.
    from modules import images

    assert hypernetwork_name, 'hypernetwork not selected'

    path = shared.hypernetworks.get(hypernetwork_name, None)
    shared.loaded_hypernetwork = Hypernetwork()
    shared.loaded_hypernetwork.load(path)

    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)
    unload = shared.opts.unload_models_when_training

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

    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=hypernetwork_name, model=shared.sd_model, device=devices.device, template_file=template_file, include_cond=True, batch_size=batch_size)
    if unload:
        shared.sd_model.cond_stage_model.to(devices.cpu)
        shared.sd_model.first_stage_model.to(devices.cpu)

    hypernetwork = shared.loaded_hypernetwork
    weights = hypernetwork.weights()
    for weight in weights:
        weight.requires_grad = True

    losses = torch.zeros((32,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"

    ititial_step = hypernetwork.step or 0
    if ititial_step > steps:
        return hypernetwork, filename

    scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)
    optimizer = torch.optim.AdamW(weights, lr=scheduler.learn_rate)

    pbar = tqdm.tqdm(enumerate(ds), total=steps - ititial_step)
    for i, entries in pbar:
        hypernetwork.step = i + ititial_step

        scheduler.apply(optimizer, hypernetwork.step)
        if scheduler.finished:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = stack_conds([entry.cond for entry in entries]).to(devices.device)
            # c = torch.vstack([entry.cond for entry in entries]).to(devices.device)
            x = torch.stack([entry.latent for entry in entries]).to(devices.device)
            loss = shared.sd_model(x, c)[0]
            del x
            del c

            losses[hypernetwork.step % losses.shape[0]] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = losses.mean()
        if torch.isnan(mean_loss):
            raise RuntimeError("Loss diverged.")
        pbar.set_description(f"loss: {mean_loss:.7f}")

        if hypernetwork.step > 0 and hypernetwork_dir is not None and hypernetwork.step % save_hypernetwork_every == 0:
            # Before saving, change name to match current checkpoint.
            hypernetwork.name = f'{hypernetwork_name}-{hypernetwork.step}'
            last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork.name}.pt')
            hypernetwork.save(last_saved_file)

        textual_inversion.write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, len(ds), {
            "loss": f"{mean_loss:.7f}",
            "learn_rate": scheduler.learn_rate
        })

        if hypernetwork.step > 0 and images_dir is not None and hypernetwork.step % create_image_every == 0:
            forced_filename = f'{hypernetwork_name}-{hypernetwork.step}'
            last_saved_image = os.path.join(images_dir, forced_filename)

            optimizer.zero_grad()
            shared.sd_model.cond_stage_model.to(devices.device)
            shared.sd_model.first_stage_model.to(devices.device)

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                do_not_save_grid=True,
                do_not_save_samples=True,
            )

            if preview_from_txt2img:
                p.prompt = preview_prompt
                p.negative_prompt = preview_negative_prompt
                p.steps = preview_steps
                p.sampler_index = preview_sampler_index
                p.cfg_scale = preview_cfg_scale
                p.seed = preview_seed
                p.width = preview_width
                p.height = preview_height
            else:
                p.prompt = entries[0].cond_text
                p.steps = 20

            preview_text = p.prompt

            processed = processing.process_images(p)
            image = processed.images[0] if len(processed.images)>0 else None

            if unload:
                shared.sd_model.cond_stage_model.to(devices.cpu)
                shared.sd_model.first_stage_model.to(devices.cpu)

            if image is not None:
                shared.state.current_image = image
                last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename)
                last_saved_image += f", prompt: {preview_text}"

        shared.state.job_no = hypernetwork.step

        shared.state.textinfo = f"""
<p>
Loss: {mean_loss:.7f}<br/>
Step: {hypernetwork.step}<br/>
Last prompt: {html.escape(entries[0].cond_text)}<br/>
Last saved hypernetwork: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    checkpoint = sd_models.select_checkpoint()

    hypernetwork.sd_checkpoint = checkpoint.hash
    hypernetwork.sd_checkpoint_name = checkpoint.model_name
    # Before saving for the last time, change name back to the base name (as opposed to the save_hypernetwork_every step-suffixed naming convention).
    hypernetwork.name = hypernetwork_name
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork.name}.pt')
    hypernetwork.save(filename)

    return hypernetwork, filename


