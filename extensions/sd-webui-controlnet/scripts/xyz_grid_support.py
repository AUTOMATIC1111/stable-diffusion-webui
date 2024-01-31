import re
import numpy as np

from modules import scripts, shared

try:
    from scripts.global_state import update_cn_models, cn_models_names, cn_preprocessor_modules
    from scripts.external_code import ResizeMode, ControlMode

except (ImportError, NameError):
    import_error = True
else:
    import_error = False

DEBUG_MODE = False


def debug_info(func):
    def debug_info_(*args, **kwargs):
        if DEBUG_MODE:
            print(f"Debug info: {func.__name__}, {args}")
        return func(*args, **kwargs)
    return debug_info_


def find_dict(dict_list, keyword, search_key="name", stop=False):
    result = next((d for d in dict_list if d[search_key] == keyword), None)
    if result or not stop:
        return result
    else:
        raise ValueError(f"Dictionary with value '{keyword}' in key '{search_key}' not found.")


def flatten(lst):
    result = []
    for element in lst:
        if isinstance(element, list):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


def is_all_included(target_list, check_list, allow_blank=False, stop=False):
    for element in flatten(target_list):
        if allow_blank and str(element) in ["None", ""]:
            continue
        elif element not in check_list:
            if not stop:
                return False
            else:
                raise ValueError(f"'{element}' is not included in check list.")
    return True


class ListParser():
    """This class restores a broken list caused by the following process
    in the xyz_grid module.
        -> valslist = [x.strip() for x in chain.from_iterable(
                                            csv.reader(StringIO(vals)))]
    It also performs type conversion,
    adjusts the number of elements in the list, and other operations.

    This class directly modifies the received list.
    """
    numeric_pattern = {
        int: {
            "range": r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*",
            "count": r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*"
        },
        float: {
            "range": r"\s*([+-]?\s*\d+(?:\.\d*)?)\s*-\s*([+-]?\s*\d+(?:\.\d*)?)(?:\s*\(([+-]\d+(?:\.\d*)?)\s*\))?\s*",
            "count": r"\s*([+-]?\s*\d+(?:\.\d*)?)\s*-\s*([+-]?\s*\d+(?:\.\d*)?)(?:\s*\[(\d+(?:\.\d*)?)\s*\])?\s*"
        }
    }

    ################################################
    #
    # Initialization method from here.
    #
    ################################################

    def __init__(self, my_list, converter=None, allow_blank=True, exclude_list=None, run=True):
        self.my_list = my_list
        self.converter = converter
        self.allow_blank = allow_blank
        self.exclude_list = exclude_list
        self.re_bracket_start = None
        self.re_bracket_start_precheck = None
        self.re_bracket_end = None
        self.re_bracket_end_precheck = None
        self.re_range = None
        self.re_count = None
        self.compile_regex()
        if run:
            self.auto_normalize()

    def compile_regex(self):
        exclude_pattern = "|".join(self.exclude_list) if self.exclude_list else None
        if exclude_pattern is None:
            self.re_bracket_start = re.compile(r"^\[")
            self.re_bracket_end = re.compile(r"\]$")
        else:
            self.re_bracket_start = re.compile(fr"^\[(?!(?:{exclude_pattern})\])")
            self.re_bracket_end = re.compile(fr"(?<!\[(?:{exclude_pattern}))\]$")

        if self.converter not in self.numeric_pattern:
            return self
        # If the converter is either int or float.
        self.re_range = re.compile(self.numeric_pattern[self.converter]["range"])
        self.re_count = re.compile(self.numeric_pattern[self.converter]["count"])
        self.re_bracket_start_precheck = None
        self.re_bracket_end_precheck = self.re_count
        return self

    ################################################
    #
    # Public method from here.
    #
    ################################################

    ################################################
    # This method is executed at the time of initialization.
    #
    def auto_normalize(self):
        if not self.has_list_notation():
            self.numeric_range_parser()
            self.type_convert()
            return self
        else:
            self.fix_structure()
            self.numeric_range_parser()
            self.type_convert()
            self.fill_to_longest()
            return self

    def has_list_notation(self):
        return any(self._search_bracket(s) for s in self.my_list)

    def numeric_range_parser(self, my_list=None, depth=0):
        if self.converter not in self.numeric_pattern:
            return self

        my_list = self.my_list if my_list is None else my_list
        result = []
        is_matched = False
        for s in my_list:
            if isinstance(s, list):
                result.extend(self.numeric_range_parser(s, depth+1))
                continue

            match = self._numeric_range_to_list(s)
            if s != match:
                is_matched = True
                result.extend(match if not depth else [match])
                continue
            else:
                result.append(s)
                continue

        if depth:
            return self._transpose(result) if is_matched else [result]
        else:
            my_list[:] = result
            return self

    def type_convert(self, my_list=None):
        my_list = self.my_list if my_list is None else my_list
        for i, s in enumerate(my_list):
            if isinstance(s, list):
                self.type_convert(s)
            elif self.allow_blank and (str(s) in ["None", ""]):
                my_list[i] = None
            elif self.converter:
                my_list[i] = self.converter(s)
            else:
                my_list[i] = s
        return self

    def fix_structure(self):
        def is_same_length(list1, list2):
            return len(list1) == len(list2)

        start_indices, end_indices = [], []
        for i, s in enumerate(self.my_list):
            if is_same_length(start_indices, end_indices):
                replace_string = self._search_bracket(s, "[", replace="")
                if s != replace_string:
                    s = replace_string
                    start_indices.append(i)
            if not is_same_length(start_indices, end_indices):
                replace_string = self._search_bracket(s, "]", replace="")
                if s != replace_string:
                    s = replace_string
                    end_indices.append(i + 1)
            self.my_list[i] = s
        if not is_same_length(start_indices, end_indices):
            raise ValueError(f"Lengths of {start_indices} and {end_indices} are different.")
        # Restore the structure of a list.
        for i, j in zip(reversed(start_indices), reversed(end_indices)):
            self.my_list[i:j] = [self.my_list[i:j]]
        return self

    def fill_to_longest(self, my_list=None, value=None, index=None):
        my_list = self.my_list if my_list is None else my_list
        if not self.sublist_exists(my_list):
            return self
        max_length = max(len(sub_list) for sub_list in my_list if isinstance(sub_list, list))
        for i, sub_list in enumerate(my_list):
            if isinstance(sub_list, list):
                fill_value = value if index is None else sub_list[index]
                my_list[i] = sub_list + [fill_value] * (max_length-len(sub_list))
        return self

    def sublist_exists(self, my_list=None):
        my_list = self.my_list if my_list is None else my_list
        return any(isinstance(item, list) for item in my_list)

    def all_sublists(self, my_list=None):    # Unused method
        my_list = self.my_list if my_list is None else my_list
        return all(isinstance(item, list) for item in my_list)

    def get_list(self):                      # Unused method
        return self.my_list

    ################################################
    #
    # Private method from here.
    #
    ################################################

    def _search_bracket(self, string, bracket="[", replace=None):
        if bracket == "[":
            pattern = self.re_bracket_start
            precheck = self.re_bracket_start_precheck  # None
        elif bracket == "]":
            pattern = self.re_bracket_end
            precheck = self.re_bracket_end_precheck
        else:
            raise ValueError(f"Invalid argument provided. (bracket: {bracket})")

        if precheck and precheck.fullmatch(string):
            return None if replace is None else string
        elif replace is None:
            return pattern.search(string)
        else:
            return pattern.sub(replace, string)

    def _numeric_range_to_list(self, string):
        match = self.re_range.fullmatch(string)
        if match is not None:
            if self.converter == int:
                start = int(match.group(1))
                end = int(match.group(2)) + 1
                step = int(match.group(3)) if match.group(3) is not None else 1
                return list(range(start, end, step))
            else:              # float
                start = float(match.group(1))
                end = float(match.group(2))
                step = float(match.group(3)) if match.group(3) is not None else 1
                return np.arange(start, end + step, step).tolist()

        match = self.re_count.fullmatch(string)
        if match is not None:
            if self.converter == int:
                start = int(match.group(1))
                end = int(match.group(2))
                num = int(match.group(3)) if match.group(3) is not None else 1
                return [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
            else:              # float
                start = float(match.group(1))
                end = float(match.group(2))
                num = int(match.group(3)) if match.group(3) is not None else 1
                return np.linspace(start=start, stop=end, num=num).tolist()
        return string

    def _transpose(self, my_list=None):
        my_list = self.my_list if my_list is None else my_list
        my_list = [item if isinstance(item, list) else [item] for item in my_list]
        self.fill_to_longest(my_list, index=-1)
        return np.array(my_list, dtype=object).T.tolist()

    ################################################
    #
    # The methods of ListParser class end here.
    #
    ################################################

################################################################
################################################################
#
# Starting the main process of this module.
#
# functions are executed in this order:
    # find_module
    # add_axis_options
    # identity
    # enable_script_control
    # apply_field
    # confirm
    # bool_
    # choices_for
    # make_excluded_list
# config lists for AxisOptions:
    # validation_data
    # extra_axis_options
################################################################
################################################################


def find_module(module_names):
    if isinstance(module_names, str):
        module_names = [s.strip() for s in module_names.split(",")]
    for data in scripts.scripts_data:
        if data.script_class.__module__ in module_names and hasattr(data, "module"):
            return data.module
    return None


def add_axis_options(xyz_grid):

    ################################################
    #
    # Define a function to pass to the AxisOption class from here.
    #
    ################################################
  
    ################################################
    # Set this function as the type attribute of the AxisOption class.
    # To skip the following processing of xyz_grid module.
    #   -> valslist = [opt.type(x) for x in valslist]
    # Perform type conversion using the function
    # set to the confirm attribute instead.
    #
    def identity(x):
        return x
 
    def enable_script_control():
        shared.opts.data["control_net_allow_script_control"] = True

    def apply_field(field):
        @debug_info
        def apply_field_(p, x, xs):
            enable_script_control()
            setattr(p, field, x)

        return apply_field_

    ################################################
    # The confirm function defined in this module
    # enables list notation and performs type conversion.
    #
    # Example:
    #     any = [any, any, any, ...]
    #     [any] = [any, None, None, ...]
    #     [None, None, any] = [None, None, any]
    #     [,,any] = [None, None, any]
    #     any, [,any,] = [any, any, any, ...], [None, any, None]
    #
    #     Enabled Only:
    #         any = [any] = [any, None, None, ...]
    #         (any and [any] are considered equivalent)
    #
    def confirm(func_or_str):
        @debug_info
        def confirm_(p, xs):
            if callable(func_or_str):           # func_or_str is converter
                ListParser(xs, func_or_str, allow_blank=True)
                return

            elif isinstance(func_or_str, str):  # func_or_str is keyword
                valid_data = find_dict(validation_data, func_or_str, stop=True)
                converter = valid_data["type"]
                exclude_list = valid_data["exclude"]() if valid_data["exclude"] else None
                check_list = valid_data["check"]()

                ListParser(xs, converter, allow_blank=True, exclude_list=exclude_list)
                is_all_included(xs, check_list, allow_blank=True, stop=True)
                return

            else:
                raise TypeError(f"Argument must be callable or str, not {type(func_or_str).__name__}.")

        return confirm_

    def bool_(string):
        string = str(string)
        if string in ["None", ""]:
            return None
        elif string.lower() in ["true", "1"]:
            return True
        elif string.lower() in ["false", "0"]:
            return False
        else:
            raise ValueError(f"Could not convert string to boolean: {string}")

    def choices_bool():
        return ["False", "True"]

    def choices_model():
        update_cn_models()
        return list(cn_models_names.values())

    def choices_control_mode():
        return [e.value for e in ControlMode]

    def choices_resize_mode():
        return [e.value for e in ResizeMode]

    def choices_preprocessor():
        return list(cn_preprocessor_modules)

    def make_excluded_list():
        pattern = re.compile(r"\[(\w+)\]")
        return [match.group(1) for s in choices_model()
                for match in pattern.finditer(s)]

    validation_data = [
        {"name": "model", "type": str, "check": choices_model, "exclude": make_excluded_list},
        {"name": "control_mode", "type": str, "check": choices_control_mode, "exclude": None},
        {"name": "resize_mode", "type": str, "check": choices_resize_mode, "exclude": None},
        {"name": "preprocessor", "type": str, "check": choices_preprocessor, "exclude": None},
    ]

    extra_axis_options = [
        xyz_grid.AxisOption("[ControlNet] Enabled", identity, apply_field("control_net_enabled"), confirm=confirm(bool_), choices=choices_bool),
        xyz_grid.AxisOption("[ControlNet] Model", identity, apply_field("control_net_model"), confirm=confirm("model"), choices=choices_model, cost=0.9),
        xyz_grid.AxisOption("[ControlNet] Weight", identity, apply_field("control_net_weight"), confirm=confirm(float)),
        xyz_grid.AxisOption("[ControlNet] Guidance Start", identity, apply_field("control_net_guidance_start"), confirm=confirm(float)),
        xyz_grid.AxisOption("[ControlNet] Guidance End", identity, apply_field("control_net_guidance_end"), confirm=confirm(float)),
        xyz_grid.AxisOption("[ControlNet] Control Mode", identity, apply_field("control_net_control_mode"), confirm=confirm("control_mode"), choices=choices_control_mode),
        xyz_grid.AxisOption("[ControlNet] Resize Mode", identity, apply_field("control_net_resize_mode"), confirm=confirm("resize_mode"), choices=choices_resize_mode),
        xyz_grid.AxisOption("[ControlNet] Preprocessor", identity, apply_field("control_net_module"), confirm=confirm("preprocessor"), choices=choices_preprocessor),
        xyz_grid.AxisOption("[ControlNet] Pre Resolution", identity, apply_field("control_net_pres"), confirm=confirm(int)),
        xyz_grid.AxisOption("[ControlNet] Pre Threshold A", identity, apply_field("control_net_pthr_a"), confirm=confirm(float)),
        xyz_grid.AxisOption("[ControlNet] Pre Threshold B", identity, apply_field("control_net_pthr_b"), confirm=confirm(float)),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)


def run():
    xyz_grid = find_module("xyz_grid.py, xy_grid.py")
    if xyz_grid:
        add_axis_options(xyz_grid)


if not import_error:
    run()
