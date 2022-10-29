import csv
import datetime
import glob
import html
import os
import sys
import traceback
import inspect

import modules.textual_inversion.dataset
import torch
import tqdm
from einops import rearrange, repeat
from ldm.util import default
from modules import devices, processing, sd_models, shared
from modules.textual_inversion import textual_inversion
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from torch import einsum
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_

from collections import defaultdict, deque
from statistics import stdev, mean


class HypernetworkModule(torch.nn.Module):
    multiplier = 1.0
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'})

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal', add_layer_norm=False, use_dropout=False):
        super().__init__()

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):

            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            # Add an activation func
            if activation_func == "linear" or activation_func is None:
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Add dropout expect last layer
            if use_dropout and i < len(layer_structure) - 3:
                linears.append(torch.nn.Dropout(p=0.3))

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    w, b = layer.weight.data, layer.bias.data
                    if weight_init == "Normal" or type(layer) == torch.nn.LayerNorm:
                        normal_(w, mean=0.0, std=0.01)
                        normal_(b, mean=0.0, std=0.005)
                    elif weight_init == 'XavierUniform':
                        xavier_uniform_(w)
                        zeros_(b)
                    elif weight_init == 'XavierNormal':
                        xavier_normal_(w)
                        zeros_(b)
                    elif weight_init == 'KaimingUniform':
                        kaiming_uniform_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    elif weight_init == 'KaimingNormal':
                        kaiming_normal_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    else:
                        raise KeyError(f"Key {weight_init} is not defined as initialization!")
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

    def __init__(self, name=None, enable_sizes=None, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.layer_structure = layer_structure
        self.activation_func = activation_func
        self.weight_init = weight_init
        self.add_layer_norm = add_layer_norm
        self.use_dropout = use_dropout

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init, self.add_layer_norm, self.use_dropout),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init, self.add_layer_norm, self.use_dropout),
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
        state_dict['activation_func'] = self.activation_func
        state_dict['is_layer_norm'] = self.add_layer_norm
        state_dict['weight_initialization'] = self.weight_init
        state_dict['use_dropout'] = self.use_dropout
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name

        torch.save(state_dict, filename)

    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cpu')

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        print(self.layer_structure)
        self.activation_func = state_dict.get('activation_func', None)
        print(f"Activation function is {self.activation_func}")
        self.weight_init = state_dict.get('weight_initialization', 'Normal')
        print(f"Weight initialization is {self.weight_init}")
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        print(f"Layer norm is set to {self.add_layer_norm}")
        self.use_dropout = state_dict.get('use_dropout', False)
        print(f"Dropout usage is set to {self.use_dropout}" )

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    HypernetworkModule(size, sd[0], self.layer_structure, self.activation_func, self.weight_init, self.add_layer_norm, self.use_dropout),
                    HypernetworkModule(size, sd[1], self.layer_structure, self.activation_func, self.weight_init, self.add_layer_norm, self.use_dropout),
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


def statistics(data):
    if len(data) < 2:
        std = 0
    else:
        std = stdev(data)
    total_information = f"loss:{mean(data):.3f}" + u"\u00B1" + f"({std/ (len(data) ** 0.5):.3f})"
    recent_data = data[-32:]
    if len(recent_data) < 2:
        std = 0
    else:
        std = stdev(recent_data)
    recent_information = f"recent 32 loss:{mean(recent_data):.3f}" + u"\u00B1" + f"({std / (len(recent_data) ** 0.5):.3f})"
    return total_information, recent_information


def report_statistics(loss_info:dict):
    keys = sorted(loss_info.keys(), key=lambda x: sum(loss_info[x]) / len(loss_info[x]))
    for key in keys:
        try:
            print("Loss statistics for file " + key)
            info, recent = statistics(list(loss_info[key]))
            print(info)
            print(recent)
        except Exception as e:
            print(e)



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

    size = len(ds.indexes)
    loss_dict = defaultdict(lambda : deque(maxlen = 1024))
    losses = torch.zeros((size,))
    previous_mean_losses = [0]
    previous_mean_loss = 0
    print("Mean loss of {} elements".format(size))

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"

    ititial_step = hypernetwork.step or 0
    if ititial_step > steps:
        return hypernetwork, filename

    scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)
    # if optimizer == "AdamW": or else Adam / AdamW / SGD, etc...
    optimizer = torch.optim.AdamW(weights, lr=scheduler.learn_rate)

    steps_without_grad = 0

    pbar = tqdm.tqdm(enumerate(ds), total=steps - ititial_step)
    for i, entries in pbar:
        hypernetwork.step = i + ititial_step
        if len(loss_dict) > 0:
            previous_mean_losses = [i[-1] for i in loss_dict.values()]
            previous_mean_loss = mean(previous_mean_losses)
            
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
            for entry in entries:
                loss_dict[entry.filename].append(loss.item())
                
            optimizer.zero_grad()
            weights[0].grad = None
            loss.backward()

            if weights[0].grad is None:
                steps_without_grad += 1
            else:
                steps_without_grad = 0
            assert steps_without_grad < 10, 'no gradient found for the trained weight after backward() for 10 steps in a row; this is a bug; training cannot continue'

            optimizer.step()

        steps_done = hypernetwork.step + 1

        if torch.isnan(losses[hypernetwork.step % losses.shape[0]]): 
            raise RuntimeError("Loss diverged.")
        
        if len(previous_mean_losses) > 1:
            std = stdev(previous_mean_losses)
        else:
            std = 0
        dataset_loss_info = f"dataset loss:{mean(previous_mean_losses):.3f}" + u"\u00B1" + f"({std / (len(previous_mean_losses) ** 0.5):.3f})"
        pbar.set_description(dataset_loss_info)

        if hypernetwork_dir is not None and steps_done % save_hypernetwork_every == 0:
            # Before saving, change name to match current checkpoint.
            hypernetwork.name = f'{hypernetwork_name}-{steps_done}'
            last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork.name}.pt')
            hypernetwork.save(last_saved_file)

        textual_inversion.write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, len(ds), {
            "loss": f"{previous_mean_loss:.7f}",
            "learn_rate": scheduler.learn_rate
        })

        if images_dir is not None and steps_done % create_image_every == 0:
            forced_filename = f'{hypernetwork_name}-{steps_done}'
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
                last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
                last_saved_image += f", prompt: {preview_text}"

        shared.state.job_no = hypernetwork.step

        shared.state.textinfo = f"""
<p>
Loss: {previous_mean_loss:.7f}<br/>
Step: {hypernetwork.step}<br/>
Last prompt: {html.escape(entries[0].cond_text)}<br/>
Last saved hypernetwork: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
        
    report_statistics(loss_dict)
    checkpoint = sd_models.select_checkpoint()

    hypernetwork.sd_checkpoint = checkpoint.hash
    hypernetwork.sd_checkpoint_name = checkpoint.model_name
    # Before saving for the last time, change name back to the base name (as opposed to the save_hypernetwork_every step-suffixed naming convention).
    hypernetwork.name = hypernetwork_name
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork.name}.pt')
    hypernetwork.save(filename)

    return hypernetwork, filename
