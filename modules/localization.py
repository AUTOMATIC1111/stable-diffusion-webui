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

        localizations[fn] = [os.path.join(dirname, file)]

    for file in scripts.list_scripts("localizations", ".json"):
        fn, ext = os.path.splitext(file.filename)
        if fn not in localizations:
            localizations[fn] = []
        localizations[fn].append(file.path)


def localization_js(current_localization_name: str) -> str:
    fns = localizations.get(current_localization_name, None)
    data = {}
    if fns is not None:
        for fn in fns:
            try:
                with open(fn, "r", encoding="utf8") as file:
                    data.update(json.load(file))
            except Exception:
                errors.report(f"Error loading localization from {fn}", exc_info=True)

    return f"window.localization = {json.dumps(data)}"
