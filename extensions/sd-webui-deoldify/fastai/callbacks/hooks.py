"Hooks provide extensibility at the model level."
from ..torch_core import *
from ..callback import *
from ..basic_train import *
from ..basic_data import *

__all__ = ['ActivationStats', 'Hook', 'HookCallback', 'Hooks', 'hook_output', 'hook_outputs',
           'model_sizes', 'num_features_model', 'model_summary', 'dummy_eval', 'dummy_batch']

class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m:nn.Module, hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hook_func,self.detach,self.stored = hook_func,detach,None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module:nn.Module, input:Tensors, output:Tensors):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()

class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."
    def __init__(self, ms:Collection[nn.Module], hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]

    def __getitem__(self,i:int)->Hook: return self.hooks[i]
    def __len__(self)->int: return len(self.hooks)
    def __iter__(self): return iter(self.hooks)
    @property
    def stored(self): return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks: h.remove()

    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()

def _hook_inner(m,i,o): return o if isinstance(o,Tensor) else o if is_listy(o) else list(o)

def hook_output (module:nn.Module, detach:bool=True, grad:bool=False)->Hook:
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, is_forward=not grad)

def hook_outputs(modules:Collection[nn.Module], detach:bool=True, grad:bool=False)->Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)

class HookCallback(LearnerCallback):
    "Callback that can be used to register hooks on `modules`. Implement the corresponding function in `self.hook`."
    def __init__(self, learn:Learner, modules:Sequence[nn.Module]=None, do_remove:bool=True):
        super().__init__(learn)
        self.modules,self.do_remove = modules,do_remove

    def on_train_begin(self, **kwargs):
        "Register the `Hooks` on `self.modules`."
        if not self.modules:
            self.modules = [m for m in flatten_model(self.learn.model)
                            if hasattr(m, 'weight')]
        self.hooks = Hooks(self.modules, self.hook)

    def on_train_end(self, **kwargs):
        "Remove the `Hooks`."
        if self.do_remove: self.remove()

    def remove(self): 
        if getattr(self, 'hooks', None): self.hooks.remove()
    def __del__(self): self.remove()

class ActivationStats(HookCallback):
    "Callback that record the mean and std of activations."

    def on_train_begin(self, **kwargs):
        "Initialize stats."
        super().on_train_begin(**kwargs)
        self.stats = []

    def hook(self, m:nn.Module, i:Tensors, o:Tensors)->Tuple[Rank0Tensor,Rank0Tensor]:
        "Take the mean and std of `o`."
        return o.mean().item(),o.std().item()
    def on_batch_end(self, train, **kwargs):
        "Take the stored results and puts it in `self.stats`"
        if train: self.stats.append(self.hooks.stored)
    def on_train_end(self, **kwargs):
        "Polish the final result."
        super().on_train_end(**kwargs)
        self.stats = tensor(self.stats).permute(2,1,0)

def dummy_batch(m: nn.Module, size:tuple=(64,64))->Tensor:
    "Create a dummy batch to go through `m` with `size`."
    ch_in = in_channels(m)
    return one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1.,1.)

def dummy_eval(m:nn.Module, size:tuple=(64,64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    m.eval()
    return m(dummy_batch(m, size))
    #return m.eval()(dummy_batch(m, size))

def model_sizes(m:nn.Module, size:tuple=(64,64))->Tuple[Sizes,Tensor,Hooks]:
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        x = dummy_eval(m, size)
        return [o.stored.shape for o in hooks]

def num_features_model(m:nn.Module)->int:
    "Return the number of output features for `model`."
    sz = 64
    while True:
        try: return model_sizes(m, size=(sz,sz))[-1][1]
        except Exception as e:
            sz *= 2
            if sz > 2048: raise

def total_params(m:nn.Module)->int:
    params, trainable = 0, False
    if hasattr(m, "weight") and hasattr(m.weight, "size"):
         params += m.weight.numel()
         trainable = m.weight.requires_grad
    if hasattr(m, "bias") and hasattr(m.bias, "size"): params += m.bias.numel()
    return params, trainable

def hook_params(modules:Collection[nn.Module])->Hooks:
    return Hooks(modules, lambda m, i, o: total_params(m))

def params_size(m: Union[nn.Module,Learner], size: tuple = (3, 64, 64))->Tuple[Sizes, Tensor, Hooks]:
    "Pass a dummy input through the model to get the various sizes. Returns (res,x,hooks) if `full`"
    if isinstance(m, Learner):
        if m.data.is_empty:
            raise Exception("This is an empty `Learner` and `Learner.summary` requires some data to pass through the model.")
        ds_type = DatasetType.Train if m.data.train_dl else (DatasetType.Valid if m.data.valid_dl else DatasetType.Test)
        x = m.data.one_batch(ds_type=ds_type, detach=False, denorm=False)[0]
        x = [o[:1] for o in x]  if is_listy(x) else x[:1]
        m = m.model
    elif isinstance(m, nn.Module): x = next(m.parameters()).new(1, *size)
    else: raise TypeError('You should either pass in a Learner or nn.Module')
    with hook_outputs(flatten_model(m)) as hook_o:
        with hook_params(flatten_model(m))as hook_p:
            x = m.eval()(*x) if is_listy(x) else m.eval()(x)
            output_size = [((o.stored.shape[1:]) if o.stored is not None else None) for o in hook_o]
            params = [(o.stored if o.stored is not None else (None,None)) for o in hook_p]
    params, trainables = map(list,zip(*params))
    return output_size, params, trainables

def get_layer_name(layer:nn.Module)->str:
    return str(layer.__class__).split(".")[-1].split("'")[0]

def layers_info(m:Collection[nn.Module]) -> Collection[namedtuple]:
    func = lambda m:list(map(get_layer_name, flatten_model(m)))
    layers_names = func(m.model) if isinstance(m, Learner) else func(m)
    layers_sizes, layers_params, layers_trainable = params_size(m)
    layer_info = namedtuple('Layer_Information', ['Layer', 'OutputSize', 'Params', 'Trainable'])
    return list(map(layer_info, layers_names, layers_sizes, layers_params, layers_trainable))

def model_summary(m:Learner, n:int=70):
    "Print a summary of `m` using a output text width of `n` chars"
    info = layers_info(m)
    header = ["Layer (type)", "Output Shape", "Param #", "Trainable"]
    res = m.model.__class__.__name__ + "\n"
    res += "=" * n + "\n"
    res += f"{header[0]:<20} {header[1]:<20} {header[2]:<10} {header[3]:<10}\n"
    res += "=" * n + "\n"
    total_params = 0
    total_trainable_params = 0
    for layer, size, params, trainable in info:
        if size is None: continue
        total_params += int(params)
        total_trainable_params += int(params) * trainable
        size, trainable = str(list(size)), str(trainable)
        res += f"{layer:<20} {size:<20} {int(params):<10,} {trainable:<10}\n"
        res += "_" * n + "\n"
    res += f"\nTotal params: {total_params:,}\n"
    res += f"Total trainable params: {total_trainable_params:,}\n"
    res += f"Total non-trainable params: {total_params - total_trainable_params:,}\n"
           
    res += f"Optimized with {str(m.opt_func)[25:-1].replace('>', '')}\n"
    if m.true_wd: res += f"Using true weight decay as discussed in https://www.fast.ai/2018/07/02/adam-weight-decay/ \n"
    if "wd" in str(m.opt_func) or "weight_decay" in str(m.opt_func): res += f"\x1b[1;31m Specifying weight decay in the optimizer has no effect, Learner will overwrite \x1b[0m \n"
    if "lr" in str(m.opt_func) or "learning_rate" in str(m.opt_func): res += f"\x1b[1;31m Specifying lr in the optimizer has no effect, pass it to fit or the defaults.lr will apply \x1b[0m \n" 
    res += f"Loss function : {m.loss_func.__class__.__name__}\n"
    res += "=" * n + "\n"
    res += "Callbacks functions applied \n"
    res += "\n".join([f"    {cbs.__class__.__name__}" for cbs in m.callbacks])

    return PrettyString(res)

Learner.summary = model_summary
