import csv
import os.path
from collections import namedtuple

PromptStyle = namedtuple("PromptStyle", ["name", "text"])


def load_styles(filename):
    res = {"None": PromptStyle("None", "")}

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf8", newline='') as file:
            reader = csv.DictReader(file)

            for row in reader:
                res[row["name"]] = PromptStyle(row["name"], row["text"])

    return res


def apply_style_text(style_text, prompt):
    if style_text == "":
        return prompt

    return prompt + ", " + style_text if prompt else style_text


def apply_style(p, style):
    if type(p.prompt) == list:
        p.prompt = [apply_style_text(style.text, x) for x in p.prompt]
    else:
        p.prompt = apply_style_text(style.text, p.prompt)


def save_style(filename, style):
    with open(filename, "a", encoding="utf8", newline='') as file:
        atstart = file.tell() == 0

        writer = csv.DictWriter(file, fieldnames=["name", "text"])

        if atstart:
            writer.writeheader()

        writer.writerow({"name": style.name, "text": style.text})
