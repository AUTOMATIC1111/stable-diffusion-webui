# NAI compatible

import torch


class HypernetworkModule(torch.nn.Module):
  def __init__(self, dim, multiplier=1.0):
    super().__init__()

    linear1 = torch.nn.Linear(dim, dim * 2)
    linear2 = torch.nn.Linear(dim * 2, dim)
    linear1.weight.data.normal_(mean=0.0, std=0.01)
    linear1.bias.data.zero_()
    linear2.weight.data.normal_(mean=0.0, std=0.01)
    linear2.bias.data.zero_()
    linears = [linear1, linear2]

    self.linear = torch.nn.Sequential(*linears)
    self.multiplier = multiplier

  def forward(self, x):
    return x + self.linear(x) * self.multiplier


class Hypernetwork(torch.nn.Module):
  enable_sizes = [320, 640, 768, 1280]
  # return self.modules[Hypernetwork.enable_sizes.index(size)]

  def __init__(self, multiplier=1.0) -> None:
    super().__init__()
    self.modules = []
    for size in Hypernetwork.enable_sizes:
      self.modules.append((HypernetworkModule(size, multiplier), HypernetworkModule(size, multiplier)))
      self.register_module(f"{size}_0", self.modules[-1][0])
      self.register_module(f"{size}_1", self.modules[-1][1])

  def apply_to_stable_diffusion(self, text_encoder, vae, unet):
    blocks = unet.input_blocks + [unet.middle_block] + unet.output_blocks
    for block in blocks:
      for subblk in block:
        if 'SpatialTransformer' in str(type(subblk)):
          for tf_block in subblk.transformer_blocks:
            for attn in [tf_block.attn1, tf_block.attn2]:
              size = attn.context_dim
              if size in Hypernetwork.enable_sizes:
                attn.hypernetwork = self
              else:
                attn.hypernetwork = None

  def apply_to_diffusers(self, text_encoder, vae, unet):
    blocks = unet.down_blocks + [unet.mid_block] + unet.up_blocks
    for block in blocks:
      if hasattr(block, 'attentions'):
        for subblk in block.attentions:
          if 'SpatialTransformer' in str(type(subblk)) or 'Transformer2DModel' in str(type(subblk)):      # 0.6.0 and 0.7~
            for tf_block in subblk.transformer_blocks:
              for attn in [tf_block.attn1, tf_block.attn2]:
                size = attn.to_k.in_features
                if size in Hypernetwork.enable_sizes:
                  attn.hypernetwork = self
                else:
                  attn.hypernetwork = None
    return True       # TODO error checking

  def forward(self, x, context):
    size = context.shape[-1]
    assert size in Hypernetwork.enable_sizes
    module = self.modules[Hypernetwork.enable_sizes.index(size)]
    return module[0].forward(context), module[1].forward(context)

  def load_from_state_dict(self, state_dict):
    # old ver to new ver
    changes = {
        'linear1.bias': 'linear.0.bias',
        'linear1.weight': 'linear.0.weight',
        'linear2.bias': 'linear.1.bias',
        'linear2.weight': 'linear.1.weight',
    }
    for key_from, key_to in changes.items():
      if key_from in state_dict:
        state_dict[key_to] = state_dict[key_from]
        del state_dict[key_from]

    for size, sd in state_dict.items():
      if type(size) == int:
        self.modules[Hypernetwork.enable_sizes.index(size)][0].load_state_dict(sd[0], strict=True)
        self.modules[Hypernetwork.enable_sizes.index(size)][1].load_state_dict(sd[1], strict=True)
    return True

  def get_state_dict(self):
    state_dict = {}
    for i, size in enumerate(Hypernetwork.enable_sizes):
      sd0 = self.modules[i][0].state_dict()
      sd1 = self.modules[i][1].state_dict()
      state_dict[size] = [sd0, sd1]
    return state_dict
