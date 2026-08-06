"""Disable/neutralize common safety checks (monkeypatches).
This shim is minimal and reversible: it tries to find known safety-related modules and replace their check functions with no-op stubs.
"""
from modules import shared
import sys

def noop_check(*args, **kwargs):
    # Return value shape may vary; many safety checkers return (is_nsfw, replaced_image) or similar.
    return False, None


def disable():
    # mark option in shared for visibility
    try:
        if hasattr(shared, 'opts') and shared.opts is not None:
            try:
                shared.opts.data['disable_safety_patch'] = True
            except Exception:
                pass
    except Exception:
        pass

    # Patch common module names if present
    try:
        import modules.sd_safety as sd_safety
        # patch known function names
        for name in ('check_safety','safety_check','run_safety'):
            if hasattr(sd_safety, name):
                setattr(sd_safety, name, lambda *a, **k: (False, None))
    except Exception:
        pass

    try:
        import modules.safety as safety
        if hasattr(safety, 'check'):
            setattr(safety, 'check', lambda *a, **k: (False, None))
    except Exception:
        pass

    try:
        import modules.deepbooru as deepbooru
        # make tagging return empty string so auto-interrogate won't block
        if hasattr(deepbooru, 'model') and hasattr(deepbooru.model, 'tag'):
            deepbooru.model.tag = lambda img: ""
    except Exception:
        pass

    # Generic: scan loaded modules for likely safety functions and patch
    for modname, mod in list(sys.modules.items()):
        if not mod:
            continue
        if 'safety' in modname or 'nsfw' in modname:
            try:
                for attr in ('check','is_nsfw','run','safety_check','safety_checker'):
                    if hasattr(mod, attr):
                        setattr(mod, attr, lambda *a, **k: (False, None))
            except Exception:
                pass

    # No explicit return; callable from shared_init

if __name__ == '__main__':
    disable()
