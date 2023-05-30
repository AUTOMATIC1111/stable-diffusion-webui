"""Utility function for gradio/external.py"""

import base64
import math
import operator
import re
import warnings
from typing import Dict, List, Tuple

import requests
import yaml

from gradio import components

##################
# Helper functions for processing tabular data
##################


def get_tabular_examples(model_name: str) -> Dict[str, List[float]]:
    readme = requests.get(f"https://huggingface.co/{model_name}/resolve/main/README.md")
    if readme.status_code != 200:
        warnings.warn(f"Cannot load examples from README for {model_name}", UserWarning)
        example_data = {}
    else:
        yaml_regex = re.search(
            "(?:^|[\r\n])---[\n\r]+([\\S\\s]*?)[\n\r]+---([\n\r]|$)", readme.text
        )
        if yaml_regex is None:
            example_data = {}
        else:
            example_yaml = next(
                yaml.safe_load_all(readme.text[: yaml_regex.span()[-1]])
            )
            example_data = example_yaml.get("widget", {}).get("structuredData", {})
    if not example_data:
        raise ValueError(
            f"No example data found in README.md of {model_name} - Cannot build gradio demo. "
            "See the README.md here: https://huggingface.co/scikit-learn/tabular-playground/blob/main/README.md "
            "for a reference on how to provide example data to your model."
        )
    # replace nan with string NaN for inference API
    for data in example_data.values():
        for i, val in enumerate(data):
            if isinstance(val, float) and math.isnan(val):
                data[i] = "NaN"
    return example_data


def cols_to_rows(
    example_data: Dict[str, List[float]]
) -> Tuple[List[str], List[List[float]]]:
    headers = list(example_data.keys())
    n_rows = max(len(example_data[header] or []) for header in headers)
    data = []
    for row_index in range(n_rows):
        row_data = []
        for header in headers:
            col = example_data[header] or []
            if row_index >= len(col):
                row_data.append("NaN")
            else:
                row_data.append(col[row_index])
        data.append(row_data)
    return headers, data


def rows_to_cols(incoming_data: Dict) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    data_column_wise = {}
    for i, header in enumerate(incoming_data["headers"]):
        data_column_wise[header] = [str(row[i]) for row in incoming_data["data"]]
    return {"inputs": {"data": data_column_wise}}


##################
# Helper functions for processing other kinds of data
##################


def postprocess_label(scores: Dict) -> Dict:
    sorted_pred = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    return {
        "label": sorted_pred[0][0],
        "confidences": [
            {"label": pred[0], "confidence": pred[1]} for pred in sorted_pred
        ],
    }


def encode_to_base64(r: requests.Response) -> str:
    # Handles the different ways HF API returns the prediction
    base64_repr = base64.b64encode(r.content).decode("utf-8")
    data_prefix = ";base64,"
    # Case 1: base64 representation already includes data prefix
    if data_prefix in base64_repr:
        return base64_repr
    else:
        content_type = r.headers.get("content-type")
        # Case 2: the data prefix is a key in the response
        if content_type == "application/json":
            try:
                data = r.json()[0]
                content_type = data["content-type"]
                base64_repr = data["blob"]
            except KeyError as ke:
                raise ValueError(
                    "Cannot determine content type returned by external API."
                ) from ke
        # Case 3: the data prefix is included in the response headers
        else:
            pass
        new_base64 = f"data:{content_type};base64,{base64_repr}"
        return new_base64


##################
# Helper function for cleaning up an Interface loaded from HF Spaces
##################


def streamline_spaces_interface(config: Dict) -> Dict:
    """Streamlines the interface config dictionary to remove unnecessary keys."""
    config["inputs"] = [
        components.get_component_instance(component)
        for component in config["input_components"]
    ]
    config["outputs"] = [
        components.get_component_instance(component)
        for component in config["output_components"]
    ]
    parameters = {
        "article",
        "description",
        "flagging_options",
        "inputs",
        "outputs",
        "title",
    }
    config = {k: config[k] for k in parameters}
    return config
