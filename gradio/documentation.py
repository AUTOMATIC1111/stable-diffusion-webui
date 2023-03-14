"""Contains methods that generate documentation for Gradio functions and classes."""

from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Tuple

classes_to_document = {}
documentation_group = None


def set_documentation_group(m):
    global documentation_group
    documentation_group = m
    if m not in classes_to_document:
        classes_to_document[m] = []


def document(*fns):
    """
    Defines the @document decorator which adds classes or functions to the Gradio
    documentation at www.gradio.app/docs.

    Usage examples:
    - Put @document() above a class to document the class and its constructor.
    - Put @document(fn1, fn2) above a class to also document the class methods fn1 and fn2.
    """

    def inner_doc(cls):
        global documentation_group
        classes_to_document[documentation_group].append((cls, fns))
        return cls

    return inner_doc


def document_fn(fn: Callable) -> Tuple[str, List[Dict], Dict, str | None]:
    """
    Generates documentation for any function.
    Parameters:
        fn: Function to document
    Returns:
        description: General description of fn
        parameters: A list of dicts for each parameter, storing data for the parameter name, annotation and doc
        return: A dict storing data for the returned annotation and doc
        example: Code for an example use of the fn
    """
    doc_str = inspect.getdoc(fn) or ""
    doc_lines = doc_str.split("\n")
    signature = inspect.signature(fn)
    description, parameters, returns, examples = [], {}, [], []
    mode = "description"
    for line in doc_lines:
        line = line.rstrip()
        if line == "Parameters:":
            mode = "parameter"
        elif line == "Example:":
            mode = "example"
        elif line == "Returns:":
            mode = "return"
        else:
            if mode == "description":
                description.append(line if line.strip() else "<br>")
                continue
            assert (
                line.startswith("    ") or line.strip() == ""
            ), f"Documentation format for {fn.__name__} has format error in line: {line}"
            line = line[4:]
            if mode == "parameter":
                colon_index = line.index(": ")
                assert (
                    colon_index > -1
                ), f"Documentation format for {fn.__name__} has format error in line: {line}"
                parameter = line[:colon_index]
                parameter_doc = line[colon_index + 2 :]
                parameters[parameter] = parameter_doc
            elif mode == "return":
                returns.append(line)
            elif mode == "example":
                examples.append(line)
    description_doc = " ".join(description)
    parameter_docs = []
    for param_name, param in signature.parameters.items():
        if param_name.startswith("_"):
            continue
        if param_name == "kwargs" and param_name not in parameters:
            continue
        parameter_doc = {
            "name": param_name,
            "annotation": param.annotation,
            "doc": parameters.get(param_name),
        }
        if param_name in parameters:
            del parameters[param_name]
        if param.default != inspect.Parameter.empty:
            default = param.default
            if type(default) == str:
                default = '"' + default + '"'
            if default.__class__.__module__ != "builtins":
                default = f"{default.__class__.__name__}()"
            parameter_doc["default"] = default
        elif parameter_doc["doc"] is not None and "kwargs" in parameter_doc["doc"]:
            parameter_doc["kwargs"] = True
        parameter_docs.append(parameter_doc)
    assert (
        len(parameters) == 0
    ), f"Documentation format for {fn.__name__} documents nonexistent parameters: {''.join(parameters.keys())}"
    if len(returns) == 0:
        return_docs = {}
    elif len(returns) == 1:
        return_docs = {"annotation": signature.return_annotation, "doc": returns[0]}
    else:
        return_docs = {}
        # raise ValueError("Does not support multiple returns yet.")
    examples_doc = "\n".join(examples) if len(examples) > 0 else None
    return description_doc, parameter_docs, return_docs, examples_doc


def document_cls(cls):
    doc_str = inspect.getdoc(cls)
    if doc_str is None:
        return "", {}, ""
    tags = {}
    description_lines = []
    mode = "description"
    for line in doc_str.split("\n"):
        line = line.rstrip()
        if line.endswith(":") and " " not in line:
            mode = line[:-1].lower()
            tags[mode] = []
        elif line.split(" ")[0].endswith(":") and not line.startswith("    "):
            tag = line[: line.index(":")].lower()
            value = line[line.index(":") + 2 :]
            tags[tag] = value
        else:
            if mode == "description":
                description_lines.append(line if line.strip() else "<br>")
            else:
                assert (
                    line.startswith("    ") or not line.strip()
                ), f"Documentation format for {cls.__name__} has format error in line: {line}"
                tags[mode].append(line[4:])
    if "example" in tags:
        example = "\n".join(tags["example"])
        del tags["example"]
    else:
        example = None
    for key, val in tags.items():
        if isinstance(val, list):
            tags[key] = "<br>".join(val)
    description = " ".join(description_lines).replace("\n", "<br>")
    return description, tags, example


def generate_documentation():
    documentation = {}
    for mode, class_list in classes_to_document.items():
        documentation[mode] = []
        for cls, fns in class_list:
            fn_to_document = cls if inspect.isfunction(cls) else cls.__init__
            _, parameter_doc, return_doc, _ = document_fn(fn_to_document)
            cls_description, cls_tags, cls_example = document_cls(cls)
            cls_documentation = {
                "class": cls,
                "name": cls.__name__,
                "description": cls_description,
                "tags": cls_tags,
                "parameters": parameter_doc,
                "returns": return_doc,
                "example": cls_example,
                "fns": [],
            }
            for fn_name in fns:
                fn = getattr(cls, fn_name)
                (
                    description_doc,
                    parameter_docs,
                    return_docs,
                    examples_doc,
                ) = document_fn(fn)
                cls_documentation["fns"].append(
                    {
                        "fn": fn,
                        "name": fn_name,
                        "description": description_doc,
                        "tags": {},
                        "parameters": parameter_docs,
                        "returns": return_docs,
                        "example": examples_doc,
                    }
                )
            documentation[mode].append(cls_documentation)
    return documentation
