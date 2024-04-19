from __future__ import annotations
from pathlib import Path
import csv
import os
import typing
import shutil


class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str | None
    negative_prompt: str | None
    path: str | None = None


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


def extract_style_text_from_prompt(style_text, prompt):
    """This function extracts the text from a given prompt based on a provided style text. It checks if the style text contains the placeholder {prompt} or if it appears at the end of the prompt. If a match is found, it returns True along with the extracted text. Otherwise, it returns False and the original prompt.

    extract_style_text_from_prompt("masterpiece", "1girl, art by greg, masterpiece") outputs (True, "1girl, art by greg")
    extract_style_text_from_prompt("masterpiece, {prompt}", "masterpiece, 1girl, art by greg") outputs (True, "1girl, art by greg")
    extract_style_text_from_prompt("masterpiece, {prompt}", "exquisite, 1girl, art by greg") outputs (False, "exquisite, 1girl, art by greg")
    """

    stripped_prompt = prompt.strip()
    stripped_style_text = style_text.strip()

    if "{prompt}" in stripped_style_text:
        left, _, right = stripped_style_text.partition("{prompt}")
        if stripped_prompt.startswith(left) and stripped_prompt.endswith(right):
            prompt = stripped_prompt[len(left) : len(stripped_prompt) - len(right)]
            return True, prompt
    else:
        if stripped_prompt.endswith(stripped_style_text):
            prompt = stripped_prompt[: len(stripped_prompt) - len(stripped_style_text)]

            if prompt.endswith(", "):
                prompt = prompt[:-2]

            return True, prompt

    return False, prompt


def extract_original_prompts(style: PromptStyle, prompt, negative_prompt):
    """
    Takes a style and compares it to the prompt and negative prompt. If the style
    matches, returns True plus the prompt and negative prompt with the style text
    removed. Otherwise, returns False with the original prompt and negative prompt.
    """
    if not style.prompt and not style.negative_prompt:
        return False, prompt, negative_prompt

    match_positive, extracted_positive = extract_style_text_from_prompt(
        style.prompt, prompt
    )
    if not match_positive:
        return False, prompt, negative_prompt

    match_negative, extracted_negative = extract_style_text_from_prompt(
        style.negative_prompt, negative_prompt
    )
    if not match_negative:
        return False, prompt, negative_prompt

    return True, extracted_positive, extracted_negative


def _format_divider(file: str) -> str:
    """
    Creates a divider for the style list.
    """
    half_len = round(len(file) / 2)
    divider = f"{'-' * (20 - half_len)} {file.upper()}"
    divider = f"{divider} {'-' * (40 - len(divider))}"
    return divider


def _expand_path(path: list[str | Path] | str | Path) -> list[str]:
    if isinstance(path, (str, Path)):
        return [str(Path(path))]

    paths = []
    for pattern in path:
        folder, file = os.path.split(pattern)
        if "*" in file or "?" in file:
            matching_files = Path(folder).glob(file)
            [paths.append(str(file)) for file in matching_files]
        else:
            paths.append(str(Path(pattern)))

    return paths


class StyleDatabase:
    def __init__(self, path: str | Path):
        self.no_style = PromptStyle("None", "", "", None)
        self.styles = {}
        self.path = path
        self.prompt_fields = [field for field in PromptStyle._fields if field != "path"]
        self.reload()

    def reload(self):
        """
        Clears the style database and reloads the styles from the CSV file(s)
        matching the path used to initialize the database.
        """
        self.styles.clear()

        # Expand the path to a list of full paths, expanding any wildcards. The
        # default path will be the first of these:
        style_files = _expand_path(self.path)
        self.default_path = style_files[0]

        for file in style_files:
            _, filename = os.path.split(file)
            # Add a visible divider to the style list
            divider = _format_divider(filename)
            self.styles[divider] = PromptStyle(f"{divider}", None, None, "do_not_save")
            # Add styles from this CSV file
            self.load_from_csv(file)

    def load_from_csv(self, path: str):
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, skipinitialspace=True)
            for row in reader:
                # Ignore empty rows or rows starting with a comment
                if not row or row["name"].startswith("#"):
                    continue
                # Support loading old CSV format with "name, text"-columns
                prompt = row["prompt"] if "prompt" in row else row["text"]
                negative_prompt = row.get("negative_prompt", "")
                # Add style to database
                self.styles[row["name"]] = PromptStyle(
                    row["name"], prompt, negative_prompt, path
                )

    def get_style_paths(self) -> set:
        """
        Using the collection of styles in the StyleDatabase, returns a set of
        all distinct files that styles are loaded from.
        """
        # Update any styles without a path to the default path
        for style in list(self.styles.values()):
            if not style.path:
                self.styles[style.name] = style._replace(path=str(self.default_path))

        # Create a list of all distinct paths, including the default path
        style_paths = set()
        style_paths.add(str(self.default_path))
        for _, style in self.styles.items():
            if style.path:
                style_paths.add(style.path)

        # Remove any paths for styles that are just list dividers
        style_paths.discard("do_not_save")

        return style_paths

    def get_style_prompts(self, styles):
        return [self.styles.get(x, self.no_style).prompt for x in styles]

    def get_negative_style_prompts(self, styles):
        return [self.styles.get(x, self.no_style).negative_prompt for x in styles]

    def apply_styles_to_prompt(self, prompt, styles):
        return apply_styles_to_prompt(
            prompt, [self.styles.get(x, self.no_style).prompt for x in styles]
        )

    def apply_negative_styles_to_prompt(self, prompt, styles):
        return apply_styles_to_prompt(
            prompt, [self.styles.get(x, self.no_style).negative_prompt for x in styles]
        )

    def save_styles(self, path: str = None) -> None:
        # The path argument is deprecated, but kept for backwards compatibility

        style_paths = self.get_style_paths()

        csv_names = [os.path.split(path)[1].lower() for path in style_paths]

        for style_path in style_paths:
            # Always keep a backup file around
            if os.path.exists(style_path):
                shutil.copy(style_path, f"{style_path}.bak")

            # Write the styles to the CSV file
            with open(style_path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.prompt_fields)
                writer.writeheader()
                for style in (s for s in self.styles.values() if s.path == style_path):
                    # Skip style list divider entries, e.g. "## STYLES.CSV ##"
                    if style.name.lower().strip("# ") in csv_names:
                        continue
                    # Write style fields, ignoring the path field
                    writer.writerow(
                        {k: v for k, v in style._asdict().items() if k != "path"}
                    )

        # Reloading the styles to re-order the drop-down lists
        self.reload()

    def extract_styles_from_prompt(self, prompt, negative_prompt):
        extracted = []

        applicable_styles = list(self.styles.values())

        while True:
            found_style = None

            for style in applicable_styles:
                is_match, new_prompt, new_neg_prompt = extract_original_prompts(
                    style, prompt, negative_prompt
                )
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
