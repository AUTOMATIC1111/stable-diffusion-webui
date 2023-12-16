import json
import sys
from dataclasses import dataclass

import gradio as gr

from modules import errors
from modules.shared_cmd_options import cmd_opts


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None, comment_before='', comment_after='', infotext=None, restrict_api=False, category_id=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.category_id = category_id
        self.refresh = refresh
        self.do_not_save = False

        self.comment_before = comment_before
        """HTML text that will be added after label in UI"""

        self.comment_after = comment_after
        """HTML text that will be added before label in UI"""

        self.infotext = infotext

        self.restrict_api = restrict_api
        """If True, the setting will not be accessible via API"""

    def link(self, label, url):
        self.comment_before += f"[<a href='{url}' target='_blank'>{label}</a>]"
        return self

    def js(self, label, js_func):
        self.comment_before += f"[<a onclick='{js_func}(); return false'>{label}</a>]"
        return self

    def info(self, info):
        self.comment_after += f"<span class='info'>({info})</span>"
        return self

    def html(self, html):
        self.comment_after += html
        return self

    def needs_restart(self):
        self.comment_after += " <span class='info'>(requires restart)</span>"
        return self

    def needs_reload_ui(self):
        self.comment_after += " <span class='info'>(requires Reload UI)</span>"
        return self


class OptionHTML(OptionInfo):
    def __init__(self, text):
        super().__init__(str(text).strip(), label='', component=lambda **kwargs: gr.HTML(elem_classes="settings-info", **kwargs))

        self.do_not_save = True


def options_section(section_identifier, options_dict):
    for v in options_dict.values():
        if len(section_identifier) == 2:
            v.section = section_identifier
        elif len(section_identifier) == 3:
            v.section = section_identifier[0:2]
            v.category_id = section_identifier[2]

    return options_dict


options_builtin_fields = {"data_labels", "data", "restricted_opts", "typemap"}


class Options:
    typemap = {int: float}

    def __init__(self, data_labels: dict[str, OptionInfo], restricted_opts):
        self.data_labels = data_labels
        self.data = {k: v.default for k, v in self.data_labels.items() if not v.do_not_save}
        self.restricted_opts = restricted_opts

    def __setattr__(self, key, value):
        if key in options_builtin_fields:
            return super(Options, self).__setattr__(key, value)

        if self.data is not None:
            if key in self.data or key in self.data_labels:
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                info = self.data_labels.get(key, None)
                if info.do_not_save:
                    return

                comp_args = info.component_args if info else None
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                if cmd_opts.hide_ui_dir_config and key in self.restricted_opts:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                self.data[key] = value
                return

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if item in options_builtin_fields:
            return super(Options, self).__getattribute__(item)

        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def set(self, key, value, is_api=False, run_callbacks=True):
        """sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""

        oldval = self.data.get(key, None)
        if oldval == value:
            return False

        option = self.data_labels[key]
        if option.do_not_save:
            return False

        if is_api and option.restrict_api:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        if run_callbacks and option.onchange is not None:
            try:
                option.onchange()
            except Exception as e:
                errors.display(e, f"changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    def get_default(self, key):
        """returns the default value for the key"""

        data_label = self.data_labels.get(key)
        if data_label is None:
            return None

        return data_label.default

    def save(self, filename):
        assert not cmd_opts.freeze_settings, "saving settings is disabled"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)

        # 1.6.0 VAE defaults
        if self.data.get('sd_vae_as_default') is not None and self.data.get('sd_vae_overrides_per_model_preferences') is None:
            self.data['sd_vae_overrides_per_model_preferences'] = not self.data.get('sd_vae_as_default')

        # 1.1.1 quicksettings list migration
        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]

        # 1.4.0 ui_reorder
        if isinstance(self.data.get('ui_reorder'), str) and self.data.get('ui_reorder') and "ui_reorder_list" not in self.data:
            self.data['ui_reorder_list'] = [i.strip() for i in self.data.get('ui_reorder').split(',')]

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func

        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, v.default) for k, v in self.data_labels.items()}
        d["_comments_before"] = {k: v.comment_before for k, v in self.data_labels.items() if v.comment_before is not None}
        d["_comments_after"] = {k: v.comment_after for k, v in self.data_labels.items() if v.comment_after is not None}

        item_categories = {}
        for item in self.data_labels.values():
            category = categories.mapping.get(item.category_id)
            category = "Uncategorized" if category is None else category.label
            if category not in item_categories:
                item_categories[category] = item.section[1]

        # _categories is a list of pairs: [section, category]. Each section (a setting page) will get a special heading above it with the category as text.
        d["_categories"] = [[v, k] for k, v in item_categories.items()] + [["Defaults", "Other"]]

        return json.dumps(d)

    def add_option(self, key, info):
        self.data_labels[key] = info
        if key not in self.data and not info.do_not_save:
            self.data[key] = info.default

    def reorder(self):
        """Reorder settings so that:
            - all items related to section always go together
            - all sections belonging to a category go together
            - sections inside a category are ordered alphabetically
            - categories are ordered by creation order

        Category is a superset of sections: for category "postprocessing" there could be multiple sections: "face restoration", "upscaling".

        This function also changes items' category_id so that all items belonging to a section have the same category_id.
        """

        category_ids = {}
        section_categories = {}

        settings_items = self.data_labels.items()
        for _, item in settings_items:
            if item.section not in section_categories:
                section_categories[item.section] = item.category_id

        for _, item in settings_items:
            item.category_id = section_categories.get(item.section)

        for category_id in categories.mapping:
            if category_id not in category_ids:
                category_ids[category_id] = len(category_ids)

        def sort_key(x):
            item: OptionInfo = x[1]
            category_order = category_ids.get(item.category_id, len(category_ids))
            section_order = item.section[1]

            return category_order, section_order

        self.data_labels = dict(sorted(settings_items, key=sort_key))

    def cast_value(self, key, value):
        """casts an arbitrary to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """

        if value is None:
            return None

        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None

        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        else:
            value = expected_type(value)

        return value


@dataclass
class OptionsCategory:
    id: str
    label: str

class OptionsCategories:
    def __init__(self):
        self.mapping = {}

    def register_category(self, category_id, label):
        if category_id in self.mapping:
            return category_id

        self.mapping[category_id] = OptionsCategory(category_id, label)


categories = OptionsCategories()
