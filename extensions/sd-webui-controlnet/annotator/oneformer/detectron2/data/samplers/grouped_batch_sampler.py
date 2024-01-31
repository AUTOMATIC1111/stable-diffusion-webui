# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from torch.utils.data.sampler import BatchSampler, Sampler


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, group_ids, batch_size):
        """
        Args:
            sampler (Sampler): Base sampler.
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}

    def __iter__(self):
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]  # yield a copy of the list
                del group_buffer[:]

    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")
