# ==========================================================
# Modified from mmcv
# ==========================================================
import ast
import os
import os.path as osp
import shutil
import sys
import tempfile
from argparse import Action
from importlib import import_module

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
RESERVED_KEYS = ["filename", "text", "pretty_text", "get", "dump", "merge_from_dict"]


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no " f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class SLConfig(object):
    """
    config files.
    only support .py file as config now.

    ref: mmcv.utils.config

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename) as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError:
            raise SyntaxError("There are syntax errors in config " f"file {filename}")

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        if filename.lower().endswith(".py"):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py")
                temp_config_name = osp.basename(temp_config_file.name)
                if os.name == 'nt':
                    temp_config_file.close()
                shutil.copyfile(filename, osp.join(temp_config_dir, temp_config_name))
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                SLConfig._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value for name, value in mod.__dict__.items() if not name.startswith("__")
                }
                # delete imported module
                del sys.modules[temp_module_name]
                # close temp file
                temp_config_file.close()
        elif filename.lower().endswith((".yml", ".yaml", ".json")):
            from .slio import slload

            cfg_dict = slload(filename)
        else:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        cfg_text = filename + "\n"
        with open(filename, "r") as f:
            cfg_text += f.read()

        # parse the base file
        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = SLConfig._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError("Duplicate key is not allowed among bases")
                    # TODO Allow the duplicate key while warnning user
                base_cfg_dict.update(c)

            base_cfg_dict = SLConfig._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b):
        """merge dict `a` into dict `b` (non-inplace).
            values in `a` will overwrite `b`.
            copy first to avoid inplace modification

        Args:
            a ([type]): [description]
            b ([type]): [description]

        Returns:
            [dict]: [description]
        """
        # import ipdb; ipdb.set_trace()
        if not isinstance(a, dict):
            return a

        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):

                if not isinstance(b[k], dict) and not isinstance(b[k], list):
                    # if :
                    # import ipdb; ipdb.set_trace()
                    raise TypeError(
                        f"{k}={v} in child config cannot inherit from base "
                        f"because {k} is a dict in the child config but is of "
                        f"type {type(b[k])} in base config. You may set "
                        f"`{DELETE_KEY}=True` to ignore the base config"
                    )
                b[k] = SLConfig._merge_a_into_b(v, b[k])
            elif isinstance(b, list):
                try:
                    _ = int(k)
                except:
                    raise TypeError(
                        f"b is a list, " f"index {k} should be an int when input but {type(k)}"
                    )
                b[int(k)] = SLConfig._merge_a_into_b(v, b[int(k)])
            else:
                b[k] = v

        return b

    @staticmethod
    def fromfile(filename):
        cfg_dict, cfg_text = SLConfig._file2dict(filename)
        return SLConfig(cfg_dict, cfg_text=cfg_text, filename=filename)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        super(SLConfig, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(SLConfig, self).__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, "r") as f:
                text = f.read()
        else:
            text = ""
        super(SLConfig, self).__setattr__("_text", text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(
                    f"dict({_indent(_format_dict(v_), indent)})," for v_ in v
                ).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{str(k)}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style="pep8",
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
        )
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        # # debug
        # print('+'*15)
        # print('name=%s' % name)
        # print("addr:", id(self))
        # # print('type(self):', type(self))
        # print(self.__dict__)
        # print('+'*15)
        # if self.__dict__ == {}:
        #     raise ValueError

        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def dump(self, file=None):
        # import ipdb; ipdb.set_trace()
        if file is None:
            return self.pretty_text
        else:
            with open(file, "w") as f:
                f.write(self.pretty_text)

    def merge_from_dict(self, options):
        """Merge list into cfg_dict

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

        Args:
            options (dict): dict of configs to merge from.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(SLConfig, self).__getattribute__("_cfg_dict")
        super(SLConfig, self).__setattr__(
            "_cfg_dict", SLConfig._merge_a_into_b(option_cfg_dict, cfg_dict)
        )

    # for multiprocess
    def __setstate__(self, state):
        self.__init__(state)

    def copy(self):
        return SLConfig(self._cfg_dict.copy())

    def deepcopy(self):
        return SLConfig(self._cfg_dict.deepcopy())


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val.lower() in ["none", "null"]:
            return None
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(",")]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)
