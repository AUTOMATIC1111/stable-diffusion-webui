import ldm.modules.encoders.modules
import open_clip
import torch
import transformers.utils.hub


class DisableInitialization:
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
        self.replaced = []
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
        for obj, field, original in self.replaced:
            setattr(obj, field, original)

        self.replaced.clear()

