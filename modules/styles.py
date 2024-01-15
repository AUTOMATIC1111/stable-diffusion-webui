# We need this so Python doesn't complain about the unknown StableDiffusionProcessing-typehint at runtime
from __future__ import annotations
import re
import os
import csv
import json
import time
from installer import log


class Style():
    def __init__(self, name: str, desc: str = "", prompt: str = "", negative_prompt: str = "", extra: str = "", filename: str = "", preview: str = "", mtime: float = 0):
        self.name = name
        self.description = desc
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.extra = extra
        self.filename = filename
        self.preview = preview
        self.mtime = mtime

def merge_prompts(style_prompt: str, prompt: str) -> str:
    if "{prompt}" in style_prompt:
        res = style_prompt.replace("{prompt}", prompt)
    else:
        original_prompt = prompt.strip()
        style_prompt = style_prompt.strip()
        parts = filter(None, (original_prompt, style_prompt))
        if original_prompt.endswith(","):
            res = " ".join(parts)
        else:
            res = ", ".join(parts)
    return res


def apply_styles_to_prompt(prompt, styles):
    for style in styles:
        prompt = merge_prompts(style, prompt)
    return prompt


def apply_styles_to_extra(p, style: Style):
    if style is None:
        return
    name_map = {
        'sampler': 'sampler_name',
    }
    from modules.generation_parameters_copypaste import parse_generation_parameters
    extra = parse_generation_parameters(style.extra)
    extra.pop('Prompt', None)
    extra.pop('Negative prompt', None)
    fields = []
    for k, v in extra.items():
        k = k.lower()
        k = k.replace(' ', '_')
        if k in name_map: # rename some fields
            k = name_map[k]
        if hasattr(p, k):
            orig = getattr(p, k)
            if type(orig) != type(v) and orig is not None:
                v = type(orig)(v)
            setattr(p, k, v)
            fields.append(f'{k}={v}')
    log.info(f'Applying style: name="{style.name}" extra={fields}')


class StyleDatabase:
    def __init__(self, opts):
        from modules import paths

        self.no_style = Style("None")
        self.styles = {}
        self.path = opts.styles_dir
        self.built_in = opts.extra_networks_styles
        if os.path.isfile(opts.styles_dir) or opts.styles_dir.endswith(".csv"):
            legacy_file = opts.styles_dir
            self.load_csv(legacy_file)
            opts.styles_dir = os.path.join(paths.models_path, "styles")
            self.path = opts.styles_dir
            os.makedirs(opts.styles_dir, exist_ok=True)
            self.save_styles(opts.styles_dir, verbose=True)
            log.debug(f'Migrated styles: file={legacy_file} folder={opts.styles_dir}')
            self.reload()
        if not os.path.isdir(opts.styles_dir):
            opts.styles_dir = os.path.join(paths.models_path, "styles")
            self.path = opts.styles_dir
            os.makedirs(opts.styles_dir, exist_ok=True)

    def load_style(self, fn, prefix=None):
        with open(fn, 'r', encoding='utf-8') as f:
            new_style = None
            try:
                all_styles = json.load(f)
                if type(all_styles) is dict:
                    all_styles = [all_styles]
                for style in all_styles:
                    if type(style) is not dict or "name" not in style:
                        raise ValueError('cannot parse style')
                    basename = os.path.splitext(os.path.basename(fn))[0]
                    name = re.sub(r'[\t\r\n]', '', style.get("name", basename)).strip()
                    if prefix is not None:
                        name = os.path.join(prefix, name)
                    else:
                        name = os.path.join(os.path.dirname(os.path.relpath(fn, self.path)), name)
                    new_style = Style(
                        name=name,
                        desc=style.get('description', name),
                        prompt=style.get("prompt", ""),
                        negative_prompt=style.get("negative", ""),
                        extra=style.get("extra", ""),
                        preview=style.get("preview", None),
                        filename=fn,
                        mtime=os.path.getmtime(fn),
                    )
                    self.styles[style["name"]] = new_style
            except Exception as e:
                log.error(f'Failed to load style: file={fn} error={e}')
            return new_style


    def reload(self):
        t0 = time.time()
        self.styles.clear()

        def list_folder(folder):
            import concurrent
            future_items = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for filename in os.listdir(folder):
                    fn = os.path.abspath(os.path.join(folder, filename))
                    if os.path.isfile(fn) and fn.lower().endswith(".json"):
                        future_items[executor.submit(self.load_style, fn, None)] = fn
                        # self.load_style(fn)
                    elif os.path.isdir(fn) and not fn.startswith('.'):
                        list_folder(fn)
                self.styles = dict(sorted(self.styles.items(), key=lambda style: style[1].filename))
                if self.built_in:
                    fn = os.path.join('html', 'art-styles.json')
                    future_items[executor.submit(self.load_style, fn, 'built-in')] = fn
                for future in concurrent.futures.as_completed(future_items):
                    future.result()

        list_folder(self.path)
        t1 = time.time()
        log.debug(f'Load styles: folder="{self.path}" items={len(self.styles.keys())} time={t1-t0:.2f}')

    def find_style(self, name):
        found = [style for style in self.styles.values() if style.name == name]
        return found[0] if len(found) > 0 else self.no_style

    def get_style_prompts(self, styles):
        if styles is None or not isinstance(styles, list):
            log.error(f'Invalid styles: {styles}')
            return []
        return [self.find_style(x).prompt for x in styles]

    def get_negative_style_prompts(self, styles):
        if styles is None or not isinstance(styles, list):
            log.error(f'Invalid styles: {styles}')
            return []
        return [self.find_style(x).negative_prompt for x in styles]

    def apply_styles_to_prompt(self, prompt, styles):
        if styles is None or not isinstance(styles, list):
            log.error(f'Invalid styles: {styles}')
            return prompt
        return apply_styles_to_prompt(prompt, [self.find_style(x).prompt for x in styles])

    def apply_negative_styles_to_prompt(self, prompt, styles):
        if styles is None or not isinstance(styles, list):
            log.error(f'Invalid styles: {styles}')
            return prompt
        return apply_styles_to_prompt(prompt, [self.find_style(x).negative_prompt for x in styles])

    def apply_styles_to_extra(self, p):
        if p.styles is None or not isinstance(p.styles, list):
            log.error(f'Invalid styles: {p.styles}')
            return
        for style in p.styles:
            s = self.find_style(style)
            apply_styles_to_extra(p, s)

    def save_styles(self, path, verbose=False):
        for name in list(self.styles):
            style = {
                "name": name,
                "prompt": self.styles[name].prompt,
                "negative": self.styles[name].negative_prompt,
                "extra": "",
                "preview": "",
            }
            keepcharacters = (' ','.','_')
            fn = "".join(c for c in name if c.isalnum() or c in keepcharacters).rstrip()
            fn = os.path.join(path, fn + ".json")
            try:
                with open(fn, 'w', encoding='utf-8') as f:
                    json.dump(style, f, indent=2)
                    if verbose:
                        log.debug(f'Saved style: name={name} file={fn}')
            except Exception as e:
                log.error(f'Failed to save style: name={name} file={path} error={e}')
        count = len(list(self.styles))
        if count > 0:
            log.debug(f'Saved styles: folder="{path}" items={count}')

    def load_csv(self, legacy_file):
        if not os.path.isfile(legacy_file):
            return
        with open(legacy_file, "r", encoding="utf-8-sig", newline='') as file:
            reader = csv.DictReader(file, skipinitialspace=True)
            num = 0
            for row in reader:
                try:
                    name = row["name"]
                    prompt = row["prompt"] if "prompt" in row else row["text"]
                    negative = row.get("negative_prompt", "") if "negative_prompt" in row else row.get("negative", "")
                    self.styles[name] = Style(name, desc=name, prompt=prompt, negative_prompt=negative, extra="")
                    log.debug(f'Migrated style: {self.styles[name].__dict__}')
                    num += 1
                except Exception:
                    log.error(f'Styles error: file="{legacy_file}" row={row}')
            log.info(f'Load legacy styles: file="{legacy_file}" loaded={num} created={len(list(self.styles))}')

    """
    def save_csv(self, path: str) -> None:
        import tempfile
        basedir = os.path.dirname(path)
        if basedir is not None and len(basedir) > 0:
            os.makedirs(basedir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(".csv")
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=Style._fields)
            writer.writeheader()
            writer.writerows(style._asdict() for k, style in self.styles.items())
            log.debug(f'Saved legacy styles: {path} {len(self.styles.keys())}')
        shutil.move(temp_path, path)
    """
