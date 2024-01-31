# Copyright (c) OpenMMLab. All rights reserved.
from .test import (collect_results_cpu, collect_results_gpu, multi_gpu_test,
                   single_gpu_test)

__all__ = [
    'collect_results_cpu', 'collect_results_gpu', 'multi_gpu_test',
    'single_gpu_test'
]
