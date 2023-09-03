# We need this so Python doesn't complain about the unknown StableDiffusionProcessing-typehint at runtime
from __future__ import annotations
import csv
import os
import json
import shutil
import typing
from installer import log
from modules import paths


if typing.TYPE_CHECKING:
    # Only import this when code is being type-checked, it doesn't have any effect at runtime
    from .processing import StableDiffusionProcessing


class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str
    extra: str = ""


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


class StyleDatabase:
    def __init__(self, opts):
        self.no_style = PromptStyle("None", "", "")
        self.styles = {}
        self.path = opts.styles_dir
        if os.path.isfile(opts.styles_dir):
            legacy_file = opts.styles_dir
            self.load_csv(legacy_file)
            opts.styles_dir = os.path.join(paths.models_path, "styles")
            self.path = opts.styles_dir
            self.mkdir()
            self.save_styles(opts.styles_dir, verbose=True)
            log.debug(f'Migrated styles: file={legacy_file} folder={self.path}')
        self.mkdir()
        self.reload()

    def mkdir(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path, exist_ok=True)
            log.debug(f'Created styles: folder={self.path}')

    def reload(self):
        self.styles.clear()
        for fn in os.listdir(self.path):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(self.path, fn), 'r', encoding='utf-8') as f:
                try:
                    style = json.load(f)
                    self.styles[style["name"]] = PromptStyle(style["name"], style["prompt"], style["negative"], style["extra"])
                except Exception as e:
                    log.error(f'Failed to load style: file={fn} error={e}')
        log.debug(f'Loaded styles: folder={self.path} items={len(self.styles.keys())}')

    def get_style_prompts(self, styles):
        return [self.styles.get(x, self.no_style).prompt for x in styles]

    def get_negative_style_prompts(self, styles):
        return [self.styles.get(x, self.no_style).negative_prompt for x in styles]

    def apply_styles_to_prompt(self, prompt, styles):
        return apply_styles_to_prompt(prompt, [self.styles.get(x, self.no_style).prompt for x in styles])

    def apply_negative_styles_to_prompt(self, prompt, styles):
        return apply_styles_to_prompt(prompt, [self.styles.get(x, self.no_style).negative_prompt for x in styles])

    def save_styles(self, path, verbose=False):
        for name in list(self.styles):
            style = {
                "name": name,
                "prompt": self.styles[name].prompt,
                "negative": self.styles[name].negative_prompt,
                "extra": "",
            }
            fn = os.path.join(path, name + ".json")
            try:
                with open(fn, 'w', encoding='utf-8') as f:
                    json.dump(style, f, indent=2)
                    if verbose:
                        log.debug(f'Saved style: name={name} file={fn}')
            except Exception as e:
                log.error(f'Failed to save style: name={name} file={path} error={e}')
        log.debug(f'Saved styles: {path} {len(self.styles.keys())}')

    def load_csv(self, legacy_file):
        with open(legacy_file, "r", encoding="utf-8-sig", newline='') as file:
            reader = csv.DictReader(file, skipinitialspace=True)
            for row in reader:
                try:
                    prompt = row["prompt"] if "prompt" in row else row["text"]
                    negative_prompt = row.get("negative_prompt", "")
                    self.styles[row["name"]] = PromptStyle(row["name"], prompt, negative_prompt)
                except Exception:
                    log.error(f'Styles error: file={legacy_file} row={row}')
            log.debug(f'Loaded legacy styles: file={legacy_file} items={len(self.styles.keys())}')

    def save_csv(self, path: str) -> None:
        import tempfile
        basedir = os.path.dirname(path)
        if basedir is not None and len(basedir) > 0:
            os.makedirs(basedir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(".csv")
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=PromptStyle._fields)
            writer.writeheader()
            writer.writerows(style._asdict() for k, style in self.styles.items())
            log.debug(f'Saved legacy styles: {path} {len(self.styles.keys())}')
        shutil.move(temp_path, path)
