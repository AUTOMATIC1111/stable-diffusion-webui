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

    def __enter__(self):
        def do_nothing(*args, **kwargs):
            pass

        def create_model_and_transforms_without_pretrained(*args, pretrained=None, **kwargs):
            return self.create_model_and_transforms(*args, pretrained=None, **kwargs)

        def CLIPTextModel_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
            return self.CLIPTextModel_from_pretrained(None, *model_args, config=pretrained_model_name_or_path, state_dict={}, **kwargs)

        def transformers_modeling_utils_load_pretrained_model(*args, **kwargs):
            args = args[0:3] + ('/', ) + args[4:]  # resolved_archive_file; must set it to something to prevent what seems to be a bug
            return self.transformers_modeling_utils_load_pretrained_model(*args, **kwargs)

        def transformers_utils_hub_get_file_from_cache(original, url, *args, **kwargs):

            # this file is always 404, prevent making request
            if url == 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/added_tokens.json':
                raise transformers.utils.hub.EntryNotFoundError

            try:
                return original(url, *args, local_files_only=True, **kwargs)
            except Exception as e:
                return original(url, *args, local_files_only=False, **kwargs)

        def transformers_utils_hub_get_from_cache(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_utils_hub_get_from_cache, url, *args, **kwargs)

        def transformers_tokenization_utils_base_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_tokenization_utils_base_cached_file, url, *args, **kwargs)

        def transformers_configuration_utils_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_configuration_utils_cached_file, url, *args, **kwargs)

        self.init_kaiming_uniform = torch.nn.init.kaiming_uniform_
        self.init_no_grad_normal = torch.nn.init._no_grad_normal_
        self.init_no_grad_uniform_ = torch.nn.init._no_grad_uniform_
        self.create_model_and_transforms = open_clip.create_model_and_transforms
        self.CLIPTextModel_from_pretrained = ldm.modules.encoders.modules.CLIPTextModel.from_pretrained
        self.transformers_modeling_utils_load_pretrained_model = getattr(transformers.modeling_utils.PreTrainedModel, '_load_pretrained_model', None)
        self.transformers_tokenization_utils_base_cached_file = getattr(transformers.tokenization_utils_base, 'cached_file', None)
        self.transformers_configuration_utils_cached_file = getattr(transformers.configuration_utils, 'cached_file', None)
        self.transformers_utils_hub_get_from_cache = getattr(transformers.utils.hub, 'get_from_cache', None)

        torch.nn.init.kaiming_uniform_ = do_nothing
        torch.nn.init._no_grad_normal_ = do_nothing
        torch.nn.init._no_grad_uniform_ = do_nothing
        open_clip.create_model_and_transforms = create_model_and_transforms_without_pretrained
        ldm.modules.encoders.modules.CLIPTextModel.from_pretrained = CLIPTextModel_from_pretrained
        if self.transformers_modeling_utils_load_pretrained_model is not None:
            transformers.modeling_utils.PreTrainedModel._load_pretrained_model = transformers_modeling_utils_load_pretrained_model
        if self.transformers_tokenization_utils_base_cached_file is not None:
            transformers.tokenization_utils_base.cached_file = transformers_tokenization_utils_base_cached_file
        if self.transformers_configuration_utils_cached_file is not None:
            transformers.configuration_utils.cached_file = transformers_configuration_utils_cached_file
        if self.transformers_utils_hub_get_from_cache is not None:
            transformers.utils.hub.get_from_cache = transformers_utils_hub_get_from_cache

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.nn.init.kaiming_uniform_ = self.init_kaiming_uniform
        torch.nn.init._no_grad_normal_ = self.init_no_grad_normal
        torch.nn.init._no_grad_uniform_ = self.init_no_grad_uniform_
        open_clip.create_model_and_transforms = self.create_model_and_transforms
        ldm.modules.encoders.modules.CLIPTextModel.from_pretrained = self.CLIPTextModel_from_pretrained
        if self.transformers_modeling_utils_load_pretrained_model is not None:
            transformers.modeling_utils.PreTrainedModel._load_pretrained_model = self.transformers_modeling_utils_load_pretrained_model
        if self.transformers_tokenization_utils_base_cached_file is not None:
            transformers.utils.hub.cached_file = self.transformers_tokenization_utils_base_cached_file
        if self.transformers_configuration_utils_cached_file is not None:
            transformers.utils.hub.cached_file = self.transformers_configuration_utils_cached_file
        if self.transformers_utils_hub_get_from_cache is not None:
            transformers.utils.hub.get_from_cache = self.transformers_utils_hub_get_from_cache

