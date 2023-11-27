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

def add_axis_options():
    extra_axis_options = [
        xyz_grid.AxisOption("[Hypertile] Unet First pass Enabled", str, bool_applier("hypertile_enable_unet"), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[Hypertile] Unet Second pass Enabled", str, bool_applier("hypertile_enable_unet_secondpass"), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[Hypertile] Unet Max Depth", int, int_applier("hypertile_max_depth_unet", 0, 3), choices=lambda: [str(x) for x in range(4)]),
        xyz_grid.AxisOption("[Hypertile] Unet Max Tile Size", int, int_applier("hypertile_max_tile_unet", 0, 512)),
        xyz_grid.AxisOption("[Hypertile] Unet Swap Size", int, int_applier("hypertile_swap_size_unet", 0, 64)),
        xyz_grid.AxisOption("[Hypertile] VAE Enabled", str, bool_applier("hypertile_enable_vae"), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[Hypertile] VAE Max Depth", int, int_applier("hypertile_max_depth_vae", 0, 3), choices=lambda: [str(x) for x in range(4)]),
        xyz_grid.AxisOption("[Hypertile] VAE Max Tile Size", int, int_applier("hypertile_max_tile_vae", 0, 512)),
        xyz_grid.AxisOption("[Hypertile] VAE Swap Size", int, int_applier("hypertile_swap_size_vae", 0, 64)),
    ]
    set_a = {opt.label for opt in xyz_grid.axis_options}
    set_b = {opt.label for opt in extra_axis_options}
    if set_a.intersection(set_b):
        return

    xyz_grid.axis_options.extend(extra_axis_options)
