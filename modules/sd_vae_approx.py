import os
import torch
from torch import nn
from modules import devices, paths, shared


sd_vae_approx_model = None


class VAEApprox(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, (7, 7))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 16, (3, 3))
        self.conv7 = nn.Conv2d(16, 8, (3, 3))
        self.conv8 = nn.Conv2d(8, 3, (3, 3))

    def forward(self, x):
        extra = 11
        try:
            x = nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
            x = nn.functional.pad(x, (extra, extra, extra, extra)) # pylint: disable=not-callable
            for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, ]:
                x = layer(x)
                x = nn.functional.leaky_relu(x, 0.1)
        except Exception:
            pass
        return x


def nn_approximation(sample): # Approximate NN
    global sd_vae_approx_model # pylint: disable=global-statement
    if sd_vae_approx_model is None:
        model_path = os.path.join(paths.models_path, "VAE-approx", "model.pt")
        sd_vae_approx_model = VAEApprox()
        if not os.path.exists(model_path):
            model_path = os.path.join(paths.script_path, "models", "VAE-approx", "model.pt")
        approx_weights = torch.load(model_path, map_location='cpu' if devices.device.type != 'cuda' else None)
        sd_vae_approx_model.load_state_dict(approx_weights)
        sd_vae_approx_model.eval()
        sd_vae_approx_model.to(devices.device, sample.dtype)
        shared.log.debug(f'Load VAE decode approximate: model="{model_path}"')
    try:
        in_sample = sample.to(devices.device).unsqueeze(0)
        x_sample = sd_vae_approx_model(in_sample)
        x_sample = x_sample[0]
        return x_sample
    except Exception as e:
        shared.log.error(f'Decode approximate: {e}')
        return sample


def cheap_approximation(sample): # Approximate simple
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2
    if shared.sd_model_type == "sdxl":
        simple_weights = torch.tensor([
            [0.4543,-0.2868, 0.1566,-0.4748],
            [0.5008, 0.0952, 0.2155,-0.3268],
            [0.5294, 0.1625,-0.0624,-0.3793]
        ]).reshape(3, 4, 1, 1)
        simple_bias = torch.tensor([0.1375, 0.0144, -0.0675])
    else:
        simple_weights = torch.tensor([
            [0.298, 0.187,-0.158,-0.184],
            [0.207, 0.286, 0.189,-0.271],
            [0.208, 0.173, 0.264,-0.473],
        ]).reshape(3, 4, 1, 1)
        simple_bias = None
    try:
        weights = simple_weights.to(sample.device, sample.dtype)
        bias = simple_bias.to(sample.device, sample.dtype) if simple_bias is not None else None
        x_sample = nn.functional.conv2d(sample, weights, bias) # pylint: disable=not-callable
        return x_sample
    except Exception as e:
        shared.log.error(f'Decode simple: {e}')
        return sample
