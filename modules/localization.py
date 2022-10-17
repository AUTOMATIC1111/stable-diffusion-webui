import json
import os
import sys
import traceback

localizations = {}


def list_localizations(dirname):
    localizations.clear()

    for file in os.listdir(dirname):
        fn, ext = os.path.splitext(file)
        if ext.lower() != ".json":
            continue

        localizations[fn] = os.path.join(dirname, file)


def localization_js(current_localization_name):
    fn = localizations.get(current_localization_name, None)
    data = {}
    if fn is not None:
        try:
            with open(fn, "r", encoding="utf8") as file:
                data = json.load(file)
        except Exception:
            print(f"Error loading localization from {fn}:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    return f"var localization = {json.dumps(data)}\n"
