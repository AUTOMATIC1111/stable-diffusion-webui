# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
"""
This module is used to store environmental parameters in PaddleSeg.

SEG_HOME : Root directory for storing PaddleSeg related data. Default to ~/.paddleseg.
           Users can change the default value through the SEG_HOME environment variable.
DATA_HOME : The directory to store the automatically downloaded dataset, e.g ADE20K.
PRETRAINED_MODEL_HOME : The directory to store the automatically downloaded pretrained model.
"""

import os

from paddleseg.utils import logger


def _get_user_home():
    return os.path.expanduser('~')


def _get_seg_home():
    if 'SEG_HOME' in os.environ:
        home_path = os.environ['SEG_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                logger.warning('SEG_HOME {} is a file!'.format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddleseg')


def _get_sub_home(directory):
    home = os.path.join(_get_seg_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


USER_HOME = _get_user_home()
SEG_HOME = _get_seg_home()
DATA_HOME = _get_sub_home('dataset')
TMP_HOME = _get_sub_home('tmp')
PRETRAINED_MODEL_HOME = _get_sub_home('pretrained_model')
