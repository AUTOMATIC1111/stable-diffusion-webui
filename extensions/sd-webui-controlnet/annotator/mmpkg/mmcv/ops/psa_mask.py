# Modified from https://github.com/hszhao/semseg/blob/master/lib/psa
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['psamask_forward', 'psamask_backward'])


class PSAMaskFunction(Function):

    @staticmethod
    def symbolic(g, input, psa_type, mask_size):
        return g.op(
            'mmcv::MMCVPSAMask',
            input,
            psa_type_i=psa_type,
            mask_size_i=mask_size)

    @staticmethod
    def forward(ctx, input, psa_type, mask_size):
        ctx.psa_type = psa_type
        ctx.mask_size = _pair(mask_size)
        ctx.save_for_backward(input)

        h_mask, w_mask = ctx.mask_size
        batch_size, channels, h_feature, w_feature = input.size()
        assert channels == h_mask * w_mask
        output = input.new_zeros(
            (batch_size, h_feature * w_feature, h_feature, w_feature))

        ext_module.psamask_forward(
            input,
            output,
            psa_type=psa_type,
            num_=batch_size,
            h_feature=h_feature,
            w_feature=w_feature,
            h_mask=h_mask,
            w_mask=w_mask,
            half_h_mask=(h_mask - 1) // 2,
            half_w_mask=(w_mask - 1) // 2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        psa_type = ctx.psa_type
        h_mask, w_mask = ctx.mask_size
        batch_size, channels, h_feature, w_feature = input.size()
        grad_input = grad_output.new_zeros(
            (batch_size, channels, h_feature, w_feature))
        ext_module.psamask_backward(
            grad_output,
            grad_input,
            psa_type=psa_type,
            num_=batch_size,
            h_feature=h_feature,
            w_feature=w_feature,
            h_mask=h_mask,
            w_mask=w_mask,
            half_h_mask=(h_mask - 1) // 2,
            half_w_mask=(w_mask - 1) // 2)
        return grad_input, None, None, None


psa_mask = PSAMaskFunction.apply


class PSAMask(nn.Module):

    def __init__(self, psa_type, mask_size=None):
        super(PSAMask, self).__init__()
        assert psa_type in ['collect', 'distribute']
        if psa_type == 'collect':
            psa_type_enum = 0
        else:
            psa_type_enum = 1
        self.psa_type_enum = psa_type_enum
        self.mask_size = mask_size
        self.psa_type = psa_type

    def forward(self, input):
        return psa_mask(input, self.psa_type_enum, self.mask_size)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(psa_type={self.psa_type}, '
        s += f'mask_size={self.mask_size})'
        return s
