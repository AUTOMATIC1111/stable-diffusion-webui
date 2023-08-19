import ldm.modules.encoders.modules
import open_clip
import torch
import transformers.utils.hub

from modules import shared


class ReplaceHelper:
    def __init__(self):
        self.replaced = []

    def replace(self, obj, field, func):
        original = getattr(obj, field, None)
        if original is None:
            return None

        self.replaced.append((obj, field, original))
        setattr(obj, field, func)

        return original

    def restore(self):
        for obj, field, original in self.replaced:
            setattr(obj, field, original)

        self.replaced.clear()


class DisableInitialization(ReplaceHelper):
    """
    When an object of this class enters a `with` block, it starts:
    - preventing torch's layer initialization functions from working
    - changes CLIP and OpenCLIP to not download model weights
    - changes CLIP to not make requests to check if there is a new version of a file you already have

    When it leaves the block, it reverts everything to how it was before.

    Use it like this:
    ```
    with DisableInitialization():
        do_things()
    ```
    """

    def __init__(self, disable_clip=True):
        super().__init__()
        self.disable_clip = disable_clip

    def replace(self, obj, field, func):
        original = getattr(obj, field, None)
        if original is None:
            return None

        self.replaced.append((obj, field, original))
        setattr(obj, field, func)

        return original

    def __enter__(self):
        def do_nothing(*args, **kwargs):
            pass

        def create_model_and_transforms_without_pretrained(*args, pretrained=None, **kwargs):
            return self.create_model_and_transforms(*args, pretrained=None, **kwargs)

        def CLIPTextModel_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
            res = self.CLIPTextModel_from_pretrained(None, *model_args, config=pretrained_model_name_or_path, state_dict={}, **kwargs)
            res.name_or_path = pretrained_model_name_or_path
            return res

        def transformers_modeling_utils_load_pretrained_model(*args, **kwargs):
            args = args[0:3] + ('/', ) + args[4:]  # resolved_archive_file; must set it to something to prevent what seems to be a bug
            return self.transformers_modeling_utils_load_pretrained_model(*args, **kwargs)

        def transformers_utils_hub_get_file_from_cache(original, url, *args, **kwargs):

            # this file is always 404, prevent making request
            if url == 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/added_tokens.json' or url == 'openai/clip-vit-large-patch14' and args[0] == 'added_tokens.json':
                return None

            try:
                res = original(url, *args, local_files_only=True, **kwargs)
                if res is None:
                    res = original(url, *args, local_files_only=False, **kwargs)
                return res
            except Exception:
                return original(url, *args, local_files_only=False, **kwargs)

        def transformers_utils_hub_get_from_cache(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_utils_hub_get_from_cache, url, *args, **kwargs)

        def transformers_tokenization_utils_base_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_tokenization_utils_base_cached_file, url, *args, **kwargs)

        def transformers_configuration_utils_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_configuration_utils_cached_file, url, *args, **kwargs)

        self.replace(torch.nn.init, 'kaiming_uniform_', do_nothing)
        self.replace(torch.nn.init, '_no_grad_normal_', do_nothing)
        self.replace(torch.nn.init, '_no_grad_uniform_', do_nothing)

        if self.disable_clip:
            self.create_model_and_transforms = self.replace(open_clip, 'create_model_and_transforms', create_model_and_transforms_without_pretrained)
            self.CLIPTextModel_from_pretrained = self.replace(ldm.modules.encoders.modules.CLIPTextModel, 'from_pretrained', CLIPTextModel_from_pretrained)
            self.transformers_modeling_utils_load_pretrained_model = self.replace(transformers.modeling_utils.PreTrainedModel, '_load_pretrained_model', transformers_modeling_utils_load_pretrained_model)
            self.transformers_tokenization_utils_base_cached_file = self.replace(transformers.tokenization_utils_base, 'cached_file', transformers_tokenization_utils_base_cached_file)
            self.transformers_configuration_utils_cached_file = self.replace(transformers.configuration_utils, 'cached_file', transformers_configuration_utils_cached_file)
            self.transformers_utils_hub_get_from_cache = self.replace(transformers.utils.hub, 'get_from_cache', transformers_utils_hub_get_from_cache)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()


class InitializeOnMeta(ReplaceHelper):
    """
    Context manager that causes all parameters for linear/conv2d/mha layers to be allocated on meta device,
    which results in those parameters having no values and taking no memory. model.to() will be broken and
    will need to be repaired by using LoadStateDictOnMeta below when loading params from state dict.

    Usage:
    ```
    with sd_disable_initialization.InitializeOnMeta():
        sd_model = instantiate_from_config(sd_config.model)
    ```
    """

    def __enter__(self):
        if shared.cmd_opts.disable_model_loading_ram_optimization:
            return

        def set_device(x):
            x["device"] = "meta"
            return x

        linear_init = self.replace(torch.nn.Linear, '__init__', lambda *args, **kwargs: linear_init(*args, **set_device(kwargs)))
        conv2d_init = self.replace(torch.nn.Conv2d, '__init__', lambda *args, **kwargs: conv2d_init(*args, **set_device(kwargs)))
        mha_init = self.replace(torch.nn.MultiheadAttention, '__init__', lambda *args, **kwargs: mha_init(*args, **set_device(kwargs)))
        self.replace(torch.nn.Module, 'to', lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()


class LoadStateDictOnMeta(ReplaceHelper):
    """
    Context manager that allows to read parameters from state_dict into a model that has some of its parameters in the meta device.
    As those parameters are read from state_dict, they will be deleted from it, so by the end state_dict will be mostly empty, to save memory.
    Meant to be used together with InitializeOnMeta above.

    Usage:
    ```
    with sd_disable_initialization.LoadStateDictOnMeta(state_dict):
        model.load_state_dict(state_dict, strict=False)
    ```
    """

    def __init__(self, state_dict, device, weight_dtype_conversion=None):
        super().__init__()
        self.state_dict = state_dict
        self.device = device
        self.weight_dtype_conversion = weight_dtype_conversion or {}
        self.default_dtype = self.weight_dtype_conversion.get('')

    def get_weight_dtype(self, key):
        key_first_term, _ = key.split('.', 1)
        return self.weight_dtype_conversion.get(key_first_term, self.default_dtype)

    def __enter__(self):
        if shared.cmd_opts.disable_model_loading_ram_optimization:
            return

        sd = self.state_dict
        device = self.device

        def load_from_state_dict(original, module, state_dict, prefix, *args, **kwargs):
            used_param_keys = []

            for name, param in module._parameters.items():
                if param is None:
                    continue

                key = prefix + name
                sd_param = sd.pop(key, None)
                if sd_param is not None:
                    state_dict[key] = sd_param.to(dtype=self.get_weight_dtype(key))
                    used_param_keys.append(key)

                if param.is_meta:
                    dtype = sd_param.dtype if sd_param is not None else param.dtype
                    module._parameters[name] = torch.nn.parameter.Parameter(torch.zeros_like(param, device=device, dtype=dtype), requires_grad=param.requires_grad)

            for name in module._buffers:
                key = prefix + name

                sd_param = sd.pop(key, None)
                if sd_param is not None:
                    state_dict[key] = sd_param
                    used_param_keys.append(key)

            original(module, state_dict, prefix, *args, **kwargs)

            for key in used_param_keys:
                state_dict.pop(key, None)

        def load_state_dict(original, module, state_dict, strict=True):
            """torch makes a lot of copies of the dictionary with weights, so just deleting entries from state_dict does not help
            because the same values are stored in multiple copies of the dict. The trick used here is to give torch a dict with
            all weights on meta device, i.e. deleted, and then it doesn't matter how many copies torch makes.

            In _load_from_state_dict, the correct weight will be obtained from a single dict with the right weights (sd).

            The dangerous thing about this is if _load_from_state_dict is not called, (if some exotic module overloads
            the function and does not call the original) the state dict will just fail to load because weights
            would be on the meta device.
            """

            if state_dict == sd:
                state_dict = {k: v.to(device="meta", dtype=v.dtype) for k, v in state_dict.items()}

            original(module, state_dict, strict=strict)

        module_load_state_dict = self.replace(torch.nn.Module, 'load_state_dict', lambda *args, **kwargs: load_state_dict(module_load_state_dict, *args, **kwargs))
        module_load_from_state_dict = self.replace(torch.nn.Module, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(module_load_from_state_dict, *args, **kwargs))
        linear_load_from_state_dict = self.replace(torch.nn.Linear, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(linear_load_from_state_dict, *args, **kwargs))
        conv2d_load_from_state_dict = self.replace(torch.nn.Conv2d, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(conv2d_load_from_state_dict, *args, **kwargs))
        mha_load_from_state_dict = self.replace(torch.nn.MultiheadAttention, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(mha_load_from_state_dict, *args, **kwargs))
        layer_norm_load_from_state_dict = self.replace(torch.nn.LayerNorm, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(layer_norm_load_from_state_dict, *args, **kwargs))
        group_norm_load_from_state_dict = self.replace(torch.nn.GroupNorm, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(group_norm_load_from_state_dict, *args, **kwargs))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()
