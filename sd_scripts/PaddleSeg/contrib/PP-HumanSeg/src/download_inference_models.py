# coding: utf8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))

from paddleseg.utils.download import download_file_and_uncompress

model_urls = {
    "portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax":
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax.zip",
    "portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax":
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip",
    "human_pp_humansegv1_lite_192x192_inference_model_with_softmax":
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_inference_model_with_softmax.zip",
    "human_pp_humansegv2_lite_192x192_inference_model_with_softmax":
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model_with_softmax.zip",
    "human_pp_humansegv1_mobile_192x192_inference_model_with_softmax":
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_inference_model_with_softmax.zip",
    "human_pp_humansegv2_mobile_192x192_inference_model_with_softmax":
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip",
}

if __name__ == "__main__":
    data_path = os.path.abspath("./inference_models")
    for model_name, url in model_urls.items():
        download_file_and_uncompress(
            url=url,
            savepath=data_path,
            extrapath=data_path,
            extraname=model_name)
    print("Download inference models finished.")
