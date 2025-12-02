# Copyright (c) Alibaba, Inc. and its affiliates.
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

# --------------------------------------------------------
# Most of the code here has been modified from:
#  https://github.com/modelscope/modelscope/blob/master/modelscope/utils/registry.py
# --------------------------------------------------------


class Registry(object):
    """This class is used to register some modules to registry by a repo
    name."""

    def __init__(self, name: str):
        """
        Initialization method.

        :param name: a registry repo name
        """
        self._name = name
        self._modules = {}

    @property
    def name(self):
        """
        Get name of current registry.

        :return: name of current registry.
        """
        return self._name

    @property
    def modules(self):
        """
        Get all modules in current registry.

        :return: a dict storing modules in current registry.
        """
        return self._modules

    def list(self):
        """Logging the list of module in current registry."""
        return list(self._modules.keys())

    def get(self, module_key):
        """
        Get module named module_key from in current registry. If not found,
        return None.

        :param module_key: specified module name
        :return: module named module_key
        """
        return self._modules.get(module_key, None)

    def _register_module(self, module_name=None, module_cls=None, force=False):
        """
        Register module to registry.

        :param module_name: module name
        :param module_cls: module class object
        :param force: Whether to override an existing class with the
            same name. Default: False.
        """

        if module_name is None:
            module_name = module_cls.__name__

        if module_name in self._modules and not force:
            raise KeyError(f"{module_name} is already registered in {self._name}")

        self._modules[module_name] = module_cls
        module_cls._name = module_name

    def register_module(self, module_name: str = None, module_cls: type = None, force=False):
        """
        Register module class object to registry with the specified modulename.

        :param module_name: module name
        :param module_cls: module class object
        :param force: Whether to override an existing class with
                the same name. Default: False.

        Example:
            >>> registry = Registry()
            >>> @registry.register_module()
            >>> class TextFormatter:
            >>>     pass

            >>> class TextFormatter2:
            >>>     pass
            >>> registry.register_module( module_name='text_formatter2',
                                        module_cls=TextFormatter2)
        """
        if not (module_name is None or isinstance(module_name, str)):
            raise TypeError(f"module_name must be either of None, str," f"got {type(module_name)}")
        if module_cls is not None:
            self._register_module(module_name=module_name, module_cls=module_cls, force=force)
            return module_cls

        # if module_cls is None, should return a decorator function
        def _register(module_cls):
            """
            Register module class object to registry.

            :param module_cls: module class object
            :return: module class object.
            """
            self._register_module(module_name=module_name, module_cls=module_cls, force=force)
            return module_cls

        return _register
