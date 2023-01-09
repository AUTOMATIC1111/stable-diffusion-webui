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
from modules import devices, processing, sd_models, shared, sd_samplers
from modules.textual_inversion import textual_inversion, logging
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from torch import einsum
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_

from collections import defaultdict, deque
from statistics import stdev, mean


optimizer_dict = {optim_name : cls_obj for optim_name, cls_obj in inspect.getmembers(torch.optim, inspect.isclass) if optim_name != "Optimizer"}

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

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal',
                 add_layer_norm=False, use_dropout=False, activate_output=False, last_layer_dropout=False):
        super().__init__()

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):

            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            # Add an activation func except last layer
            if activation_func == "linear" or activation_func is None or (i >= len(layer_structure) - 2 and not activate_output):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Add dropout except last layer
            if use_dropout and (i < len(layer_structure) - 3 or last_layer_dropout and i < len(layer_structure) - 2):
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
                        normal_(b, mean=0.0, std=0)
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

    def __init__(self, name=None, enable_sizes=None, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, activate_output=False, **kwargs):
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
        self.activate_output = activate_output
        self.last_layer_dropout = kwargs['last_layer_dropout'] if 'last_layer_dropout' in kwargs else True
        self.optimizer_name = None
        self.optimizer_state_dict = None

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.use_dropout, self.activate_output, last_layer_dropout=self.last_layer_dropout),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.use_dropout, self.activate_output, last_layer_dropout=self.last_layer_dropout),
            )
        self.eval_mode()

    def weights(self):
        res = []
        for k, layers in self.layers.items():
            for layer in layers:
                res += layer.parameters()
        return res

    def train_mode(self):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True

    def eval_mode(self):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def save(self, filename):
        state_dict = {}
        optimizer_saved_dict = {}

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
        state_dict['activate_output'] = self.activate_output
        state_dict['last_layer_dropout'] = self.last_layer_dropout

        if self.optimizer_name is not None:
            optimizer_saved_dict['optimizer_name'] = self.optimizer_name

        torch.save(state_dict, filename)
        if shared.opts.save_optimizer_state and self.optimizer_state_dict:
            optimizer_saved_dict['hash'] = sd_models.model_hash(filename)
            optimizer_saved_dict['optimizer_state_dict'] = self.optimizer_state_dict
            torch.save(optimizer_saved_dict, filename + '.optim')

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
        self.activate_output = state_dict.get('activate_output', True)
        print(f"Activate last layer is set to {self.activate_output}")
        self.last_layer_dropout = state_dict.get('last_layer_dropout', False)

        optimizer_saved_dict = torch.load(self.filename + '.optim', map_location = 'cpu') if os.path.exists(self.filename + '.optim') else {}
        self.optimizer_name = optimizer_saved_dict.get('optimizer_name', 'AdamW')
        print(f"Optimizer name is {self.optimizer_name}")
        if sd_models.model_hash(filename) == optimizer_saved_dict.get('hash', None):
            self.optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
        else:
            self.optimizer_state_dict = None
        if self.optimizer_state_dict:
            print("Loaded existing optimizer from checkpoint")
        else:
            print("No saved optimizer exists in checkpoint")

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    HypernetworkModule(size, sd[0], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.use_dropout, self.activate_output, last_layer_dropout=self.last_layer_dropout),
                    HypernetworkModule(size, sd[1], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.use_dropout, self.activate_output, last_layer_dropout=self.last_layer_dropout),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)


def list_hypernetworks(path):
    res = {}
    for filename in sorted(glob.iglob(os.path.join(path, '**/*.pt'), recursive=True)):
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name + f"({sd_models.model_hash(filename)})"] = filename
    return res


def load_hypernetwork(filename):
    path = shared.hypernetworks.get(filename, None)
    # Prevent any file named "None.pt" from being loaded.
    if path is not None and filename != "None":
        print(f"Loading hypernetwork {filename}")
        try:
            shared.loaded_hypernetwork = Hypernetwork()
            shared.loaded_hypernetwork.load(path)

        except Exception:
            print(f"Error loading hypernetwork {path}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    else:
        if shared.loaded_hypernetwork is not None:
            print("Unloading hypernetwork")

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


def create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False):
    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))

    fn = os.path.join(shared.cmd_opts.hypernetwork_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    if type(layer_structure) == str:
        layer_structure = [float(x.strip()) for x in layer_structure.split(",")]

    hypernet = modules.hypernetworks.hypernetwork.Hypernetwork(
        name=name,
        enable_sizes=[int(x) for x in enable_sizes],
        layer_structure=layer_structure,
        activation_func=activation_func,
        weight_init=weight_init,
        add_layer_norm=add_layer_norm,
        use_dropout=use_dropout,
    )
    hypernet.save(fn)

    shared.reload_hypernetworks()


def train_hypernetwork(hypernetwork_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width, training_height, steps, clip_grad_mode, clip_grad_value, shuffle_tags, tag_drop_out, latent_sampling_method, create_image_every, save_hypernetwork_every, template_file, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    # images allows training previews to have infotext. Importing it at the top causes a circular import problem.
    from modules import images

    save_hypernetwork_every = save_hypernetwork_every or 0
    create_image_every = create_image_every or 0
    textual_inversion.validate_train_inputs(hypernetwork_name, learn_rate, batch_size, gradient_step, data_root, template_file, steps, save_hypernetwork_every, create_image_every, log_directory, name="hypernetwork")

    path = shared.hypernetworks.get(hypernetwork_name, None)
    shared.loaded_hypernetwork = Hypernetwork()
    shared.loaded_hypernetwork.load(path)

    shared.state.job = "train-hypernetwork"
    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps

    hypernetwork_name = hypernetwork_name.rsplit('(', 1)[0]
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

    hypernetwork = shared.loaded_hypernetwork
    checkpoint = sd_models.select_checkpoint()

    initial_step = hypernetwork.step or 0
    if initial_step >= steps:
        shared.state.textinfo = "Model has already been trained beyond specified max steps"
        return hypernetwork, filename

    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    
    clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else None
    if clip_grad:
        clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)

    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."

    pin_memory = shared.opts.pin_memory

    ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=hypernetwork_name, model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=template_file, include_cond=True, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method)

    if shared.opts.save_training_settings_to_txt:
        saved_params = dict(
            model_name=checkpoint.model_name, model_hash=checkpoint.hash, num_of_dataset_images=len(ds),
            **{field: getattr(hypernetwork, field) for field in ['layer_structure', 'activation_func', 'weight_init', 'add_layer_norm', 'use_dropout', ]}
        )
        logging.save_settings_to_file(log_directory, {**saved_params, **locals()})

    latent_sampling_method = ds.latent_sampling_method

    dl = modules.textual_inversion.dataset.PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)

    old_parallel_processing_allowed = shared.parallel_processing_allowed

    if unload:
        shared.parallel_processing_allowed = False
        shared.sd_model.cond_stage_model.to(devices.cpu)
        shared.sd_model.first_stage_model.to(devices.cpu)

    weights = hypernetwork.weights()
    hypernetwork.train_mode()

    # Here we use optimizer from saved HN, or we can specify as UI option.
    if hypernetwork.optimizer_name in optimizer_dict:
        optimizer = optimizer_dict[hypernetwork.optimizer_name](params=weights, lr=scheduler.learn_rate)
        optimizer_name = hypernetwork.optimizer_name
    else:
        print(f"Optimizer type {hypernetwork.optimizer_name} is not defined!")
        optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate)
        optimizer_name = 'AdamW'

    if hypernetwork.optimizer_state_dict:  # This line must be changed if Optimizer type can be different from saved optimizer.
        try:
            optimizer.load_state_dict(hypernetwork.optimizer_state_dict)
        except RuntimeError as e:
            print("Cannot resume from saved optimizer!")
            print(e)

    scaler = torch.cuda.amp.GradScaler()
    
    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # n steps = batch_size * gradient_step * n image processed
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0 #internal
    # size = len(ds.indexes)
    # loss_dict = defaultdict(lambda : deque(maxlen = 1024))
    # losses = torch.zeros((size,))
    # previous_mean_losses = [0]
    # previous_mean_loss = 0
    # print("Mean loss of {} elements".format(size))

    steps_without_grad = 0

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"

    pbar = tqdm.tqdm(total=steps - initial_step)
    try:
        for i in range((steps-initial_step) * gradient_step):
            if scheduler.finished:
                break
            if shared.state.interrupted:
                break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                scheduler.apply(optimizer, hypernetwork.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break

                if clip_grad:
                    clip_grad_sched.step(hypernetwork.step)
                
                with devices.autocast():
                    x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                    if tag_drop_out != 0 or shuffle_tags:
                        shared.sd_model.cond_stage_model.to(devices.device)
                        c = shared.sd_model.cond_stage_model(batch.cond_text).to(devices.device, non_blocking=pin_memory)
                        shared.sd_model.cond_stage_model.to(devices.cpu)
                    else:
                        c = stack_conds(batch.cond).to(devices.device, non_blocking=pin_memory)
                    loss = shared.sd_model(x, c)[0] / gradient_step
                    del x
                    del c

                    _loss_step += loss.item()
                scaler.scale(loss).backward()
                
                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue

                if clip_grad:
                    clip_grad(weights, clip_grad_sched.learn_rate)
                
                scaler.step(optimizer)
                scaler.update()
                hypernetwork.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0

                steps_done = hypernetwork.step + 1
                
                epoch_num = hypernetwork.step // steps_per_epoch
                epoch_step = hypernetwork.step % steps_per_epoch

                pbar.set_description(f"[Epoch {epoch_num}: {epoch_step+1}/{steps_per_epoch}]loss: {loss_step:.7f}")
                if hypernetwork_dir is not None and steps_done % save_hypernetwork_every == 0:
                    # Before saving, change name to match current checkpoint.
                    hypernetwork_name_every = f'{hypernetwork_name}-{steps_done}'
                    last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork_name_every}.pt')
                    hypernetwork.optimizer_name = optimizer_name
                    if shared.opts.save_optimizer_state:
                        hypernetwork.optimizer_state_dict = optimizer.state_dict()
                    save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, last_saved_file)
                    hypernetwork.optimizer_state_dict = None  # dereference it after saving, to save memory.

                textual_inversion.write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, steps_per_epoch, {
                    "loss": f"{loss_step:.7f}",
                    "learn_rate": scheduler.learn_rate
                })

                if images_dir is not None and steps_done % create_image_every == 0:
                    forced_filename = f'{hypernetwork_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)
                    hypernetwork.eval_mode()
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
                        p.sampler_name = sd_samplers.samplers[preview_sampler_index].name
                        p.cfg_scale = preview_cfg_scale
                        p.seed = preview_seed
                        p.width = preview_width
                        p.height = preview_height
                    else:
                        p.prompt = batch.cond_text[0]
                        p.steps = 20
                        p.width = training_width
                        p.height = training_height

                    preview_text = p.prompt

                    processed = processing.process_images(p)
                    image = processed.images[0] if len(processed.images) > 0 else None

                    if unload:
                        shared.sd_model.cond_stage_model.to(devices.cpu)
                        shared.sd_model.first_stage_model.to(devices.cpu)
                    hypernetwork.train_mode()
                    if image is not None:
                        shared.state.current_image = image
                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"

                shared.state.job_no = hypernetwork.step

                shared.state.textinfo = f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved hypernetwork: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        pbar.leave = False
        pbar.close()
        hypernetwork.eval_mode()
        #report_statistics(loss_dict)

    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')
    hypernetwork.optimizer_name = optimizer_name
    if shared.opts.save_optimizer_state:
        hypernetwork.optimizer_state_dict = optimizer.state_dict()
    save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, filename)

    del optimizer
    hypernetwork.optimizer_state_dict = None  # dereference it after saving, to save memory.
    shared.sd_model.cond_stage_model.to(devices.device)
    shared.sd_model.first_stage_model.to(devices.device)
    shared.parallel_processing_allowed = old_parallel_processing_allowed

    return hypernetwork, filename

def save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, filename):
    old_hypernetwork_name = hypernetwork.name
    old_sd_checkpoint = hypernetwork.sd_checkpoint if hasattr(hypernetwork, "sd_checkpoint") else None
    old_sd_checkpoint_name = hypernetwork.sd_checkpoint_name if hasattr(hypernetwork, "sd_checkpoint_name") else None
    try:
        hypernetwork.sd_checkpoint = checkpoint.hash
        hypernetwork.sd_checkpoint_name = checkpoint.model_name
        hypernetwork.name = hypernetwork_name
        hypernetwork.save(filename)
    except:
        hypernetwork.sd_checkpoint = old_sd_checkpoint
        hypernetwork.sd_checkpoint_name = old_sd_checkpoint_name
        hypernetwork.name = old_hypernetwork_name
        raise
