from modules.patches import patch
from modules.errors import report
from inspect import signature
from functools import wraps

try:
    from huggingface_hub.utils import LocalEntryNotFoundError
    from huggingface_hub import file_download

    def try_local_files_only(func):
        if (param := signature(func).parameters.get('local_files_only', None)) and not param.kind == param.KEYWORD_ONLY:
            raise ValueError(f'{func.__name__} does not have keyword-only parameter "local_files_only"')

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from modules.shared import opts
                try_offline_mode = not kwargs.get('local_files_only') and opts.hd_dl_local_first
            except Exception:
                report('Error in try_local_files_only - skip try_local_files_only', exc_info=True)
                try_offline_mode = False

            if try_offline_mode:
                try:
                    return func(*args, **{**kwargs, 'local_files_only': True})
                except LocalEntryNotFoundError:
                    pass
                except Exception:
                    report('Unexpected exception in try_local_files_only - retry without patch', exc_info=True)

            return func(*args, **kwargs)

        return wrapper

    try:
        patch(__name__, file_download, 'hf_hub_download', try_local_files_only(file_download.hf_hub_download))
    except RuntimeError:
        pass  # already patched

except Exception:
    report('Error patching hf_hub_download', exc_info=True)
