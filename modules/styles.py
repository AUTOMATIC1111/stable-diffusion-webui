import csv
import fnmatch
import os
import os.path
import re
import typing
import shutil


class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str
    path: str = None


def clean_text(text: str) -> str:
    """
    Iterating through a list of regular expressions and replacement strings, we
    clean up the prompt and style text to make it easier to match against each
    other.
    """
    re_list = [
        ("multiple commas", re.compile("(,+\s+)+,?"), ", "),
        ("multiple spaces", re.compile("\s{2,}"), " "),
    ]
    for _, regex, replace in re_list:
        text = regex.sub(replace, text)

    return text.strip(", ")


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

    return clean_text(prompt)


def unwrap_style_text_from_prompt(style_text, prompt):
    """
    Checks the prompt to see if the style text is wrapped around it. If so,
    returns True plus the prompt text without the style text. Otherwise, returns
    False with the original prompt.

    Note that the "cleaned" version of the style text is only used for matching
    purposes here. It isn't returned; the original style text is not modified.
    """
    stripped_prompt = clean_text(prompt)
    stripped_style_text = clean_text(style_text)
    if "{prompt}" in stripped_style_text:
        # Work out whether the prompt is wrapped in the style text. If so, we
        # return True and the "inner" prompt text that isn't part of the style.
        try:
            left, right = stripped_style_text.split("{prompt}", 2)
        except ValueError as e:
            # If the style text has multple "{prompt}"s, we can't split it into
            # two parts. This is an error, but we can't do anything about it.
            print(f"Unable to compare style text to prompt:\n{style_text}")
            print(f"Error: {e}")
            return False, prompt
        if stripped_prompt.startswith(left) and stripped_prompt.endswith(right):
            prompt = stripped_prompt[len(left) : len(stripped_prompt) - len(right)]
            return True, prompt
    else:
        # Work out whether the given prompt ends with the style text. If so, we
        # return True and the prompt text up to where the style text starts.
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

    match_positive, extracted_positive = unwrap_style_text_from_prompt(
        style.prompt, prompt
    )
    if not match_positive:
        return False, prompt, negative_prompt

    match_negative, extracted_negative = unwrap_style_text_from_prompt(
        style.negative_prompt, negative_prompt
    )
    if not match_negative:
        return False, prompt, negative_prompt

    return True, extracted_positive, extracted_negative


class StyleDatabase:
    def __init__(self, path: str):
        self.no_style = PromptStyle("None", "", "", None)
        self.styles = {}
        self.path = path

        folder, file = os.path.split(self.path)
        self.default_file = file.split("*")[0] + ".csv"
        if self.default_file == ".csv":
            self.default_file = "styles.csv"
        self.default_path = os.path.join(folder, self.default_file)

        self.prompt_fields = [field for field in PromptStyle._fields if field != "path"]

        self.reload()

    def reload(self):
        """
        Clears the style database and reloads the styles from the CSV file(s)
        matching the path used to initialize the database.
        """
        self.styles.clear()

        path, filename = os.path.split(self.path)

        if "*" in filename:
            fileglob = filename.split("*")[0] + "*.csv"
            filelist = []
            for file in os.listdir(path):
                if fnmatch.fnmatch(file, fileglob):
                    filelist.append(file)
                    # Add a visible divider to the style list
                    half_len = round(len(file) / 2)
                    divider = f"{'-' * (20 - half_len)} {file.upper()}"
                    divider = f"{divider} {'-' * (40 - len(divider))}"
                    self.styles[divider] = PromptStyle(
                        f"{divider}", None, None, "do_not_save"
                    )
                    # Add styles from this CSV file
                    self.load_from_csv(os.path.join(path, file))
            if len(filelist) == 0:
                print(f"No styles found in {path} matching {fileglob}")
                return
        elif not os.path.exists(self.path):
            print(f"Style database not found: {self.path}")
            return
        else:
            self.load_from_csv(self.path)

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

    def get_style_paths(self) -> list():
        """
        Returns a list of all distinct paths, including the default path, of
        files that styles are loaded from."""
        # Update any styles without a path to the default path
        for style in list(self.styles.values()):
            if not style.path:
                self.styles[style.name] = style._replace(path=self.default_path)

        # Create a list of all distinct paths, including the default path
        style_paths = set()
        style_paths.add(self.default_path)
        for _, style in self.styles.items():
            if style.path:
                style_paths.add(style.path)

        # Remove any paths for styles that are just list dividers
        style_paths.remove("do_not_save")

        return list(style_paths)

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
        _ = path

        # Update any styles without a path to the default path
        for style in list(self.styles.values()):
            if not style.path:
                self.styles[style.name] = style._replace(path=self.default_path)

        # Create a list of all distinct paths, including the default path
        style_paths = set()
        style_paths.add(self.default_path)
        for _, style in self.styles.items():
            if style.path:
                style_paths.add(style.path)

        # Remove any paths for styles that are just list dividers
        style_paths.remove("do_not_save")

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
                    # Skip style list dividers, e.g. "STYLES.CSV"
                    if style.name.lower().strip("# ") in csv_names:
                        continue
                    # Write style fields, ignoring the path field
                    writer.writerow(
                        {k: v for k, v in style._asdict().items() if k != "path"}
                    )

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
