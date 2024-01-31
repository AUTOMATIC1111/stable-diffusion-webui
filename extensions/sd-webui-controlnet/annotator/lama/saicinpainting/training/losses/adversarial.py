from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAdversarialLoss:
    def pre_generator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           generator: nn.Module, discriminator: nn.Module):
        """
        Prepare for generator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        """
        Prepare for discriminator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def generator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                       discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate generator loss
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total generator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate discriminator loss and call .backward() on it
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total discriminator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def interpolate_mask(self, mask, shape):
        assert mask is not None
        assert self.allow_scale_mask or shape == mask.shape[-2:]
        if shape != mask.shape[-2:] and self.allow_scale_mask:
            if self.mask_scale_mode == 'maxpool':
                mask = F.adaptive_max_pool2d(mask, shape)
            else:
                mask = F.interpolate(mask, size=shape, mode=self.mask_scale_mode)
        return mask

def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty

class NonSaturatingWithR1(BaseAdversarialLoss):
    def __init__(self, gp_coef=5, weight=1, mask_as_fake_target=False, allow_scale_mask=False,
                 mask_scale_mode='nearest', extra_mask_weight_for_gen=0,
                 use_unmasked_for_gen=True, use_unmasked_for_discr=True):
        self.gp_coef = gp_coef
        self.weight = weight
        # use for discr => use for gen;
        # otherwise we teach only the discr to pay attention to very small difference
        assert use_unmasked_for_gen or (not use_unmasked_for_discr)
        # mask as target => use unmasked for discr:
        # if we don't care about unmasked regions at all
        # then it doesn't matter if the value of mask_as_fake_target is true or false
        assert use_unmasked_for_discr or (not mask_as_fake_target)
        self.use_unmasked_for_gen = use_unmasked_for_gen
        self.use_unmasked_for_discr = use_unmasked_for_discr
        self.mask_as_fake_target = mask_as_fake_target
        self.allow_scale_mask = allow_scale_mask
        self.mask_scale_mode = mask_scale_mode
        self.extra_mask_weight_for_gen = extra_mask_weight_for_gen

    def generator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                       discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                       mask=None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_loss = F.softplus(-discr_fake_pred)
        if (self.mask_as_fake_target and self.extra_mask_weight_for_gen > 0) or \
                not self.use_unmasked_for_gen:  # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            if not self.use_unmasked_for_gen:
                fake_loss = fake_loss * mask
            else:
                pixel_weights = 1 + mask * self.extra_mask_weight_for_gen
                fake_loss = fake_loss * pixel_weights

        return fake_loss.mean() * self.weight, dict()

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        real_batch.requires_grad = True

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                           mask=None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, real_batch) * self.gp_coef
        fake_loss = F.softplus(discr_fake_pred)

        if not self.use_unmasked_for_discr or self.mask_as_fake_target:
            # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            # use_unmasked_for_discr=False only makes sense for fakes;
            # for reals there is no difference beetween two regions
            fake_loss = fake_loss * mask
            if self.mask_as_fake_target:
                fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)

        sum_discr_loss = real_loss + grad_penalty + fake_loss
        metrics = dict(discr_real_out=discr_real_pred.mean(),
                       discr_fake_out=discr_fake_pred.mean(),
                       discr_real_gp=grad_penalty)
        return sum_discr_loss.mean(), metrics

class BCELoss(BaseAdversarialLoss):
    def __init__(self, weight):
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(self, discr_fake_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_mask_gt = torch.zeros(discr_fake_pred.shape).to(discr_fake_pred.device)
        fake_loss = self.bce_loss(discr_fake_pred, real_mask_gt) * self.weight
        return fake_loss, dict()

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        real_batch.requires_grad = True

    def discriminator_loss(self,
                           mask: torch.Tensor,
                           discr_real_pred: torch.Tensor,
                           discr_fake_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        real_mask_gt = torch.zeros(discr_real_pred.shape).to(discr_real_pred.device)
        sum_discr_loss = (self.bce_loss(discr_real_pred, real_mask_gt) +  self.bce_loss(discr_fake_pred, mask)) / 2
        metrics = dict(discr_real_out=discr_real_pred.mean(),
                       discr_fake_out=discr_fake_pred.mean(),
                       discr_real_gp=0)
        return sum_discr_loss, metrics


def make_discrim_loss(kind, **kwargs):
    if kind == 'r1':
        return NonSaturatingWithR1(**kwargs)
    elif kind == 'bce':
        return BCELoss(**kwargs)
    raise ValueError(f'Unknown adversarial loss kind {kind}')
