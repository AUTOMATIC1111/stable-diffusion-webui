import ldm.modules.encoders.modules
import open_clip
import torch


class DisableInitialization:
    """
    When an object of this class enters a `with` block, it starts preventing torch's layer initialization
    functions from working, and changes CLIP and OpenCLIP to not download model weights. When it leaves,
    reverts everything to how it was.

    Use like this:
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

        self.init_kaiming_uniform = torch.nn.init.kaiming_uniform_
        self.init_no_grad_normal = torch.nn.init._no_grad_normal_
        self.create_model_and_transforms = open_clip.create_model_and_transforms
        self.CLIPTextModel_from_pretrained = ldm.modules.encoders.modules.CLIPTextModel.from_pretrained

        torch.nn.init.kaiming_uniform_ = do_nothing
        torch.nn.init._no_grad_normal_ = do_nothing
        open_clip.create_model_and_transforms = create_model_and_transforms_without_pretrained
        ldm.modules.encoders.modules.CLIPTextModel.from_pretrained = CLIPTextModel_from_pretrained

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.nn.init.kaiming_uniform_ = self.init_kaiming_uniform
        torch.nn.init._no_grad_normal_ = self.init_no_grad_normal
        open_clip.create_model_and_transforms = self.create_model_and_transforms
        ldm.modules.encoders.modules.CLIPTextModel.from_pretrained = self.CLIPTextModel_from_pretrained

