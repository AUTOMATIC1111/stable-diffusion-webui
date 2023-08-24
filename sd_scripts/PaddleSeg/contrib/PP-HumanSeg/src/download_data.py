# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))

from paddleseg.utils.download import download_file_and_uncompress

if __name__ == "__main__":
    data_path = os.path.abspath("./data")
    urls = [
        "https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip",
        "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/videos.zip",
        "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/images.zip"
    ]

    for url in urls:
        download_file_and_uncompress(
            url=url, savepath=data_path, extrapath=data_path)

    print("Data download finished!")
