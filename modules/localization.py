import json
import os

from modules import errors, scripts

localizations = {}


def list_localizations(dirname):
    localizations.clear()

    for file in os.listdir(dirname):
        fn, ext = os.path.splitext(file)
        if ext.lower() != ".json":
            continue

        localizations[fn] = os.path.join(dirname, file)

    for file in scripts.list_scripts("localizations", ".json"):
        fn, ext = os.path.splitext(file.filename)
        localizations[fn] = file.path


def localization_js(current_localization_name: str) -> str:
    fn = localizations.get(current_localization_name, None)
    data = {}
    if fn is not None:
        try:
            with open(fn, "r", encoding="utf8") as file:
                data = json.load(file)
        except Exception:
            errors.report(f"Error loading localization from {fn}", exc_info=True)

    return f"window.localization = {json.dumps(data)}"
