import csv
import os
import os.path
import re
import typing
import shutil


class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str


def merge_prompts(style_prompt: str, prompt: str) -> str:
    if "{prompt}" in style_prompt:
        res = style_prompt.replace("{prompt}", prompt)
    else:
        parts = filter(None, (prompt.strip(), style_prompt.strip()))
        res = ", ".join(parts)

    return res


def apply_styles_to_prompt(prompt, styles):
    for style in styles:
        prompt = merge_prompts(style, prompt)

    return prompt


re_spaces = re.compile("  +")


def extract_style_text_from_prompt(style_text, prompt):
    stripped_prompt = re.sub(re_spaces, " ", prompt.strip())
    stripped_style_text = re.sub(re_spaces, " ", style_text.strip())
    if "{prompt}" in stripped_style_text:
        left, right = stripped_style_text.split("{prompt}", 2)
        if stripped_prompt.startswith(left) and stripped_prompt.endswith(right):
            prompt = stripped_prompt[len(left):len(stripped_prompt)-len(right)]
            return True, prompt
    else:
        if stripped_prompt.endswith(stripped_style_text):
            prompt = stripped_prompt[:len(stripped_prompt)-len(stripped_style_text)]

            if prompt.endswith(', '):
                prompt = prompt[:-2]

            return True, prompt

    return False, prompt


def extract_style_from_prompts(style: PromptStyle, prompt, negative_prompt):
    if not style.prompt and not style.negative_prompt:
        return False, prompt, negative_prompt

    match_positive, extracted_positive = extract_style_text_from_prompt(style.prompt, prompt)
    if not match_positive:
        return False, prompt, negative_prompt

    match_negative, extracted_negative = extract_style_text_from_prompt(style.negative_prompt, negative_prompt)
    if not match_negative:
        return False, prompt, negative_prompt

    return True, extracted_positive, extracted_negative


class StyleDatabase:
    def __init__(self, path: str):
        self.no_style = PromptStyle("None", "", "")
        self.styles = {}
        self.path = path

        self.reload()

    def reload(self):
        self.styles.clear()

        if not os.path.exists(self.path):
            return

        with open(self.path, "r", encoding="utf-8-sig", newline='') as file:
            reader = csv.DictReader(file, skipinitialspace=True)
            for row in reader:
                # Support loading old CSV format with "name, text"-columns
                prompt = row["prompt"] if "prompt" in row else row["text"]
                negative_prompt = row.get("negative_prompt", "")
                self.styles[row["name"]] = PromptStyle(row["name"], prompt, negative_prompt)

    def get_style_prompts(self, styles):
        return [self.styles.get(x, self.no_style).prompt for x in styles]

    def get_negative_style_prompts(self, styles):
        return [self.styles.get(x, self.no_style).negative_prompt for x in styles]

    def apply_styles_to_prompt(self, prompt, styles):
        return apply_styles_to_prompt(prompt, [self.styles.get(x, self.no_style).prompt for x in styles])

    def apply_negative_styles_to_prompt(self, prompt, styles):
        return apply_styles_to_prompt(prompt, [self.styles.get(x, self.no_style).negative_prompt for x in styles])

    def save_styles(self, path: str) -> None:
        # Always keep a backup file around
        if os.path.exists(path):
            shutil.copy(path, f"{path}.bak")

        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline='') as file:
            # _fields is actually part of the public API: typing.NamedTuple is a replacement for collections.NamedTuple,
            # and collections.NamedTuple has explicit documentation for accessing _fields. Same goes for _asdict()
            writer = csv.DictWriter(file, fieldnames=PromptStyle._fields)
            writer.writeheader()
            writer.writerows(style._asdict() for k, style in self.styles.items())

    def extract_styles_from_prompt(self, prompt, negative_prompt):
        extracted = []

        applicable_styles = list(self.styles.values())

        while True:
            found_style = None

            for style in applicable_styles:
                is_match, new_prompt, new_neg_prompt = extract_style_from_prompts(style, prompt, negative_prompt)
                if is_match:
                    found_style = style
                    prompt = new_prompt
                    negative_prompt = new_neg_prompt
                    break

            if not found_style:
                break

            applicable_styles.remove(found_style)
            extracted.append(found_style.name)

        return list(reversed(extracted)), prompt, negative_prompt
