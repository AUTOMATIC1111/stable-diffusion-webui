from modules import scripts
from modules.shared import opts

xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module

def int_applier(value_name:str, min_range:int = -1, max_range:int = -1):
    """
    Returns a function that applies the given value to the given value_name in opts.data.
    """
    def validate(value_name:str, value:str):
        value = int(value)
        # validate value
        if not min_range == -1:
            assert value >= min_range, f"Value {value} for {value_name} must be greater than or equal to {min_range}"
        if not max_range == -1:
            assert value <= max_range, f"Value {value} for {value_name} must be less than or equal to {max_range}"
    def apply_int(p, x, xs):
        validate(value_name, x)
        opts.data[value_name] = int(x)
    return apply_int

def bool_applier(value_name:str):
    """
    Returns a function that applies the given value to the given value_name in opts.data.
    """
    def validate(value_name:str, value:str):
        assert value.lower() in ["true", "false"], f"Value {value} for {value_name} must be either true or false"
    def apply_bool(p, x, xs):
        validate(value_name, x)
        value_boolean = x.lower() == "true"
        opts.data[value_name] = value_boolean
    return apply_bool

def float_applier(value_name:str, min_range:float = -1, max_range:float = -1):
    """
    Returns a function that applies the given value to the given value_name in opts.data.
    """
    def validate(value_name:str, value:str):
        value = float(value)
        # validate value
        if not min_range == -1:
            assert value >= min_range, f"Value {value} for {value_name} must be greater than or equal to {min_range}"
        if not max_range == -1:
            assert value <= max_range, f"Value {value} for {value_name} must be less than or equal to {max_range}"
    def apply_float(p, x, xs):
        validate(value_name, x)
        opts.data[value_name] = float(x)
    return apply_float

def add_axis_options():
    extra_axis_options = [
        xyz_grid.AxisOption("[DeepCache] Enabled", str, bool_applier("deepcache_enable"), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[DeepCache] Cache Resnet level", int, int_applier("deepcache_cache_resnet_level", 0, 10)),
        xyz_grid.AxisOption("[DeepCache] Cache Disable initial step percentage", float, float_applier("deepcache_cache_enable_step_percentage", 0, 1)),
        xyz_grid.AxisOption("[DeepCache] Cache Refresh Rate", int, int_applier("deepcache_full_run_step_rate", 0, 1000)),
        xyz_grid.AxisOption("[DeepCache] HR Reuse", str, bool_applier("deepcache_hr_reuse"), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[DeepCache] HR Cache Disable initial step percentage", float, float_applier("deepcache_cache_enable_step_percentage_hr", 0, 1)),
    ]
    set_a = {opt.label for opt in xyz_grid.axis_options}
    set_b = {opt.label for opt in extra_axis_options}
    if set_a.intersection(set_b):
        return

    xyz_grid.axis_options.extend(extra_axis_options)
