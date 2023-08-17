# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import sys
import paddle

# A global variable to record the number of calling times for profiler
# functions. It is used to specify the tracing range of training steps.
_profiler_step_id = 0

# A global variable to avoid parsing from string every time.
_profiler_options = None


class ProfilerOptions(object):
    '''
    Use a string to initialize a ProfilerOptions.
    The string should be in the format: "key1=value1;key2=value;key3=value3".
    For example:
      "profile_path=model.profile"
      "batch_range=[50, 60]; profile_path=model.profile"
      "batch_range=[50, 60]; tracer_option=OpDetail; profile_path=model.profile"
    ProfilerOptions supports following key-value pair:
      batch_range      - a integer list, e.g. [100, 110].
      state            - a string, the optional values are 'CPU', 'GPU' or 'All'.
      sorted_key       - a string, the optional values are 'calls', 'total',
                         'max', 'min' or 'ave.
      tracer_option    - a string, the optional values are 'Default', 'OpDetail',
                         'AllOpDetail'.
      profile_path     - a string, the path to save the serialized profile data,
                         which can be used to generate a timeline.
      exit_on_finished - a boolean.
    '''

    def __init__(self, options_str):
        assert isinstance(options_str, str)

        self._options = {
            'batch_range': [10, 20],
            'state': 'All',
            'sorted_key': 'total',
            'tracer_option': 'Default',
            'profile_path': '/tmp/profile',
            'exit_on_finished': True
        }

        if options_str != "":
            self._parse_from_string(options_str)

    def _parse_from_string(self, options_str):
        for kv in options_str.replace(' ', '').split(';'):
            key, value = kv.split('=')
            if key == 'batch_range':
                value_list = value.replace('[', '').replace(']', '').split(',')
                value_list = list(map(int, value_list))
                if len(value_list) >= 2 and value_list[0] >= 0 and value_list[
                        1] > value_list[0]:
                    self._options[key] = value_list
            elif key == 'exit_on_finished':
                self._options[key] = value.lower() in ("yes", "true", "t", "1")
            elif key in [
                    'state', 'sorted_key', 'tracer_option', 'profile_path'
            ]:
                self._options[key] = value

    def __getitem__(self, name):
        if self._options.get(name, None) is None:
            raise ValueError(
                "ProfilerOptions does not have an option named %s." % name)
        return self._options[name]


def add_profiler_step(options_str=None):
    '''
    Enable the operator-level timing using PaddlePaddle's profiler.
    The profiler uses a independent variable to count the profiler steps.
    One call of this function is treated as a profiler step.

    Args:
      profiler_options - a string to initialize the ProfilerOptions.
                         Default is None, and the profiler is disabled.
    '''
    if options_str is None:
        return

    global _profiler_step_id
    global _profiler_options

    if _profiler_options is None:
        _profiler_options = ProfilerOptions(options_str)

    if _profiler_step_id == _profiler_options['batch_range'][0]:
        paddle.utils.profiler.start_profiler(_profiler_options['state'],
                                             _profiler_options['tracer_option'])
    elif _profiler_step_id == _profiler_options['batch_range'][1]:
        paddle.utils.profiler.stop_profiler(_profiler_options['sorted_key'],
                                            _profiler_options['profile_path'])
        if _profiler_options['exit_on_finished']:
            sys.exit(0)

    _profiler_step_id += 1
