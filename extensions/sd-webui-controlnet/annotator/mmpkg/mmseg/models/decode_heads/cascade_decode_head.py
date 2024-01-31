from abc import ABCMeta, abstractmethod

from .decode_head import BaseDecodeHead


class BaseCascadeDecodeHead(BaseDecodeHead, metaclass=ABCMeta):
    """Base class for cascade decode head used in
    :class:`CascadeEncoderDecoder."""

    def __init__(self, *args, **kwargs):
        super(BaseCascadeDecodeHead, self).__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs, prev_output):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg,
                      train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, prev_output)
        losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, prev_output)
