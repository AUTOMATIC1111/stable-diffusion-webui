# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
from collections.abc import Sequence

import warnings


class ComponentManager:
    """
    Implement a manager class to add the new component properly.
    The component can be added as either class or function type.

    Args:
        name (str): The name of component.

    Returns:
        A callable object of ComponentManager.

    Examples 1:

        from paddleseg.cvlibs.manager import ComponentManager

        model_manager = ComponentManager()

        class AlexNet: ...
        class ResNet: ...

        model_manager.add_component(AlexNet)
        model_manager.add_component(ResNet)

        # Or pass a sequence alliteratively:
        model_manager.add_component([AlexNet, ResNet])
        print(model_manager.components_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}

    Examples 2:

        # Or an easier way, using it as a Python decorator, while just add it above the class declaration.
        from paddleseg.cvlibs.manager import ComponentManager

        model_manager = ComponentManager()

        @model_manager.add_component
        class AlexNet: ...

        @model_manager.add_component
        class ResNet: ...

        print(model_manager.components_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
    """

    def __init__(self, name=None):
        self._components_dict = dict()
        self._name = name

    def __len__(self):
        return len(self._components_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, list(self._components_dict.keys()))

    def __getitem__(self, item):
        if item not in self._components_dict.keys():
            raise KeyError("{} does not exist in availabel {}".format(item,
                                                                      self))
        return self._components_dict[item]

    @property
    def components_dict(self):
        return self._components_dict

    @property
    def name(self):
        return self._name

    def _add_single_component(self, component):
        """
        Add a single component into the corresponding manager.

        Args:
            component (function|class): A new component.

        Raises:
            TypeError: When `component` is neither class nor function.
            KeyError: When `component` was added already.
        """

        # Currently only support class or function type
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError("Expect class/function type, but received {}".
                            format(type(component)))

        # Obtain the internal name of the component
        component_name = component.__name__

        # Check whether the component was added already
        if component_name in self._components_dict.keys():
            warnings.warn("{} exists already! It is now updated to {} !!!".
                          format(component_name, component))
            self._components_dict[component_name] = component

        else:
            # Take the internal name of the component as its key
            self._components_dict[component_name] = component

    def add_component(self, components):
        """
        Add component(s) into the corresponding manager.

        Args:
            components (function|class|list|tuple): Support four types of components.

        Returns:
            components (function|class|list|tuple): Same with input components.
        """

        # Check whether the type is a sequence
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            component = components
            self._add_single_component(component)

        return components


MODELS = ComponentManager("models")
BACKBONES = ComponentManager("backbones")
DATASETS = ComponentManager("datasets")
TRANSFORMS = ComponentManager("transforms")
LOSSES = ComponentManager("losses")
OPTIMIZERS = ComponentManager("optimizers")
