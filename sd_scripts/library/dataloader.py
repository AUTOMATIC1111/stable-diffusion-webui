#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/7 9:34 AM
# @Author  : wangdongming
# @Site    : 
# @File    : dataloader.py
# @Software: xingzhe.ai
import typing
import torch
from torch.utils.data import BatchSampler, DataLoader
from accelerate.data_loader import DataLoaderDispatcher as AccelerateDataLoaderDispatcher
from accelerate.data_loader import DataLoaderShard as AccelerateDataLoaderShard, DataLoaderStateMixin
from accelerate.state import GradientState
from accelerate.utils import (
    send_to_device,
    synchronize_rng_states,
)


class DataLoaderShard(DataLoader, DataLoaderStateMixin):
    """
    Subclass of a PyTorch `DataLoader` that will deal with device placement and current distributed setup.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        device (`torch.device`, *optional*):
            If passed, the device to put all batches on.
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: an optional `torch.Generator`
        synchronized_generator (`torch.Generator`, *optional*):
            A random number generator to keep synchronized across processes.
        split_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.

    **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
    """

    def __init__(self, dataset, device=None, rng_types=None, synchronized_generator=None, skip_batches=0, **kwargs):
        super().__init__(dataset, **kwargs)
        self.device = device
        self.rng_types = rng_types
        self.synchronized_generator = synchronized_generator
        self.skip_batches = skip_batches
        self.gradient_state = GradientState()
        self.dataloader_iter = None

    def __iter__(self):
        self.begin()
        self.dataloader_iter = super().__iter__()
        print(f"set dataloader iter:{type(self.dataloader_iter)}")

        # We iterate one batch ahead to check when we are at the end
        try:
            current_batch = next(self.dataloader_iter)
        except StopIteration:
            yield

        batch_index = 0
        while True:
            try:
                # But we still move it to the device so it is done before `StopIteration` is reached
                if self.device is not None:
                    current_batch = send_to_device(current_batch, self.device)
                next_batch = next(self.dataloader_iter)
                if batch_index >= self.skip_batches:
                    yield current_batch
                batch_index += 1
                current_batch = next_batch
            except StopIteration:
                self.end_of_dataloader = True
                if batch_index >= self.skip_batches:
                    yield current_batch
                break
        self.end()

    @property
    def total_batch_size(self):
        batch_sampler = self.sampler if isinstance(self.sampler, BatchSampler) else self.batch_sampler
        return (
            batch_sampler.batch_size
            if getattr(batch_sampler, "split_batches", False)
            else (batch_sampler.batch_size * getattr(batch_sampler, "num_processes", 1))
        )

    @property
    def total_dataset_length(self):
        return getattr(self.dataset, "total_length", len(self.dataset))

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        print("dataloader shutdown...")
        if getattr(self, 'dataloader_iter', None):
            print(f"del {type(self.dataloader_iter)}")
            del self.dataloader_iter
        else:
            print("self.dataloader_iter is none")


def _convert_dataloadershard(accelerator_dataloader: AccelerateDataLoaderShard):
    dataset = accelerator_dataloader.dataset
    device = accelerator_dataloader.device
    rng_types = accelerator_dataloader.rng_types
    synchronized_generator = accelerator_dataloader.synchronized_generator
    skip_batches = accelerator_dataloader.skip_batches
    keys = [
        'batch_size',
        'shuffle', 'sampler',
        'batch_sampler',
        'num_workers',
        'collate_fn',
        'pin_memory',
        'drop_last',
        'timeout',
        'worker_init_fn',
        'multiprocessing_context',
        'generator',
        'prefetch_factor',
        'persistent_workers',
        'pin_memory_device'
    ]
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]
    kwargs = {}
    for k in keys:
        if k in ignore_kwargs:
            continue

        if hasattr(accelerator_dataloader, k):
            kwargs.update({
                k: getattr(accelerator_dataloader, k)
            })

    print("convert dataloader...")
    return DataLoaderShard(
        dataset, device, rng_types, synchronized_generator, skip_batches, **kwargs)


def convert_dataloader(dataloader: typing.Union[AccelerateDataLoaderShard, AccelerateDataLoaderDispatcher]):
    if isinstance(dataloader, AccelerateDataLoaderDispatcher):
        raise TypeError('DataLoaderDispatcher unsupported')

    return _convert_dataloadershard(dataloader)
