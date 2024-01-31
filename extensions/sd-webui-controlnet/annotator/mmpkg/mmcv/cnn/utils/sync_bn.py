import torch

import annotator.mmpkg.mmcv as mmcv


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """A general BatchNorm layer without input dimension check.

    Reproduced from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)
    The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
    is `_check_input_dim` that is designed for tensor sanity checks.
    The check has been bypassed in this class for the convenience of converting
    SyncBatchNorm.
    """

    def _check_input_dim(self, input):
        return


def revert_sync_batchnorm(module):
    """Helper function to convert all `SyncBatchNorm` (SyncBN) and
    `mmcv.ops.sync_bn.SyncBatchNorm`(MMSyncBN) layers in the model to
    `BatchNormXd` layers.

    Adapted from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.

    Returns:
        module_output: The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]
    if hasattr(mmcv, 'ops'):
        module_checklist.append(mmcv.ops.SyncBatchNorm)
    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(module.num_features, module.eps,
                                     module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            # no_grad() may not be needed here but
            # just to be consistent with `convert_sync_batchnorm()`
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        # qconfig exists in quantized models
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output
