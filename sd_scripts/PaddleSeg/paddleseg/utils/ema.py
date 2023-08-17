# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle


def judge_params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not paddle.equal_all(ema_param[1], param[1]):
            # print("Difference in", ema_param[0])
            return False
    return True


def init_ema_params(ema_model, model):
    state = {}
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        if paddle.is_floating_point(v):
            v = msd[k].detach()
        state[k] = v
    ema_model.set_state_dict(state)


def update_ema_model(ema_model, model, step=0, decay=0.999):
    with paddle.no_grad():
        state = {}
        decay = min(1 - 1 / (step + 1), decay)
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if paddle.is_floating_point(v):
                v *= decay
                v += (1.0 - decay) * msd[k].detach()
            state[k] = v
        ema_model.set_state_dict(state)