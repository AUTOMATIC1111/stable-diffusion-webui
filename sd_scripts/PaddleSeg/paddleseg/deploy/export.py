# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class WrappedModel(paddle.nn.Layer):
    def __init__(self, model, output_op):
        super().__init__()
        self.model = model
        self.output_op = output_op
        assert output_op in ['argmax', 'softmax'], \
            "output_op should in ['argmax', 'softmax']"

    def forward(self, x):
        outs = self.model(x)
        new_outs = []
        for out in outs:
            if self.output_op == 'argmax':
                out = paddle.argmax(out, axis=1, dtype='int32')
            elif self.output_op == 'softmax':
                out = paddle.nn.functional.softmax(out, axis=1)
            new_outs.append(out)
        return new_outs
