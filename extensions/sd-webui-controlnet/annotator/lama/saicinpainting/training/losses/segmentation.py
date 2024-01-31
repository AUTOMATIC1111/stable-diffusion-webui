import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import weights as constant_weights


class CrossEntropy2d(nn.Module):
    def __init__(self, reduction="mean", ignore_label=255, weights=None, *args, **kwargs):
        """
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size "nclasses"
        """
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label
        self.weights = weights
        if self.weights is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.weights = torch.FloatTensor(constant_weights[weights]).to(device)

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, 1, h, w)
        """
        target = target.long()
        assert not target.requires_grad
        assert predict.dim() == 4, "{0}".format(predict.size())
        assert target.dim() == 4, "{0}".format(target.size())
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert target.size(1) == 1, "{0}".format(target.size(1))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        target = target.squeeze(1)
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=self.weights, reduction=self.reduction)
        return loss
