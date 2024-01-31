from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.callbacks import hook_outputs
import torchvision.models as models


class FeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[20, 70, 10]):
        super().__init__()

        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
        requires_grad(self.m_feat, False)
        blocks = [
            i - 1
            for i, o in enumerate(children(self.m_feat))
            if isinstance(o, nn.MaxPool2d)
        ]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel'] + [f'feat_{i}' for i in range(len(layer_ids))]
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


# Refactored code, originally from https://github.com/VinceMarron/style_transfer
class WassFeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[5, 15, 2], wass_wgts=[3.0, 0.7, 0.01]):
        super().__init__()
        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
        requires_grad(self.m_feat, False)
        blocks = [
            i - 1
            for i, o in enumerate(children(self.m_feat))
            if isinstance(o, nn.MaxPool2d)
        ]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.wass_wgts = wass_wgts
        self.metric_names = (
            ['pixel']
            + [f'feat_{i}' for i in range(len(layer_ids))]
            + [f'wass_{i}' for i in range(len(layer_ids))]
        )
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def _calc_2_moments(self, tensor):
        chans = tensor.shape[1]
        tensor = tensor.view(1, chans, -1)
        n = tensor.shape[2]
        mu = tensor.mean(2)
        tensor = (tensor - mu[:, :, None]).squeeze(0)
        # Prevents nasty bug that happens very occassionally- divide by zero.  Why such things happen?
        if n == 0:
            return None, None
        cov = torch.mm(tensor, tensor.t()) / float(n)
        return mu, cov

    def _get_style_vals(self, tensor):
        mean, cov = self._calc_2_moments(tensor)
        if mean is None:
            return None, None, None
        eigvals, eigvects = torch.symeig(cov, eigenvectors=True)
        eigroot_mat = torch.diag(torch.sqrt(eigvals.clamp(min=0)))
        root_cov = torch.mm(torch.mm(eigvects, eigroot_mat), eigvects.t())
        tr_cov = eigvals.clamp(min=0).sum()
        return mean, tr_cov, root_cov

    def _calc_l2wass_dist(
        self, mean_stl, tr_cov_stl, root_cov_stl, mean_synth, cov_synth
    ):
        tr_cov_synth = torch.symeig(cov_synth, eigenvectors=True)[0].clamp(min=0).sum()
        mean_diff_squared = (mean_stl - mean_synth).pow(2).sum()
        cov_prod = torch.mm(torch.mm(root_cov_stl, cov_synth), root_cov_stl)
        var_overlap = torch.sqrt(
            torch.symeig(cov_prod, eigenvectors=True)[0].clamp(min=0) + 1e-8
        ).sum()
        dist = mean_diff_squared + tr_cov_stl + tr_cov_synth - 2 * var_overlap
        return dist

    def _single_wass_loss(self, pred, targ):
        mean_test, tr_cov_test, root_cov_test = targ
        mean_synth, cov_synth = self._calc_2_moments(pred)
        loss = self._calc_l2wass_dist(
            mean_test, tr_cov_test, root_cov_test, mean_synth, cov_synth
        )
        return loss

    def forward(self, input, target):
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        styles = [self._get_style_vals(i) for i in out_feat]

        if styles[0][0] is not None:
            self.feat_losses += [
                self._single_wass_loss(f_pred, f_targ) * w
                for f_pred, f_targ, w in zip(in_feat, styles, self.wass_wgts)
            ]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()
