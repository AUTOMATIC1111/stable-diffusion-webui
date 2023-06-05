"""This module should not be used directly as its API is subject to change. Instead,
use the `gr.Blocks.load()` or `gr.load()` functions."""

from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING, Callable

import requests
from gradio_client import Client
from gradio_client.documentation import document, set_documentation_group

import gradio
from gradio import components, utils
from gradio.context import Context
from gradio.exceptions import Error, TooManyRequestsError
from gradio.external_utils import (
    cols_to_rows,
    encode_to_base64,
    get_tabular_examples,
    postprocess_label,
    rows_to_cols,
    streamline_spaces_interface,
)
from gradio.processing_utils import extract_base64_data, to_binary

if TYPE_CHECKING:
    from gradio.blocks import Blocks
    from gradio.interface import Interface


set_documentation_group("helpers")


@document()
def load(
    name: str,
    src: str | None = None,
    api_key: str | None = None,
    hf_token: str | None = None,
    alias: str | None = None,
    **kwargs,
) -> Blocks:
    """
    Method that constructs a Blocks from a Hugging Face repo. Can accept
    model repos (if src is "models") or Space repos (if src is "spaces"). The input
    and output components are automatically loaded from the repo.
    Parameters:
        name: the name of the model (e.g. "gpt2" or "facebook/bart-base") or space (e.g. "flax-community/spanish-gpt2"), can include the `src` as prefix (e.g. "models/facebook/bart-base")
        src: the source of the model: `models` or `spaces` (or leave empty if source is provided as a prefix in `name`)
        api_key: Deprecated. Please use the `hf_token` parameter instead.
        hf_token: optional access token for loading private Hugging Face Hub models or spaces. Find your token here: https://huggingface.co/settings/tokens.  Warning: only provide this if you are loading a trusted private Space as it can be read by the Space you are loading.
        alias: optional string used as the name of the loaded model instead of the default name (only applies if loading a Space running Gradio 2.x)
    Returns:
        a Gradio Blocks object for the given model
    Example:
        import gradio as gr
        demo = gr.load("gradio/question-answering", src="spaces")
        demo.launch()
    """
    if hf_token is None and api_key:
        warnings.warn(
            "The `api_key` parameter will be deprecated. Please use the `hf_token` parameter going forward."
        )
        hf_token = api_key
    return load_blocks_from_repo(
        name=name, src=src, api_key=hf_token, alias=alias, **kwargs
    )


def load_blocks_from_repo(
    name: str,
    src: str | None = None,
    api_key: str | None = None,
    alias: str | None = None,
    **kwargs,
) -> Blocks:
    """Creates and returns a Blocks instance from a Hugging Face model or Space repo."""
    if src is None:
        # Separate the repo type (e.g. "model") from repo name (e.g. "google/vit-base-patch16-224")
        tokens = name.split("/")
        assert (
            len(tokens) > 1
        ), "Either `src` parameter must be provided, or `name` must be formatted as {src}/{repo name}"
        src = tokens[0]
        name = "/".join(tokens[1:])

    factory_methods: dict[str, Callable] = {
        # for each repo type, we have a method that returns the Interface given the model name & optionally an api_key
        "huggingface": from_model,
        "models": from_model,
        "spaces": from_spaces,
    }
    assert (
        src.lower() in factory_methods
    ), f"parameter: src must be one of {factory_methods.keys()}"

    if api_key is not None:
        if Context.hf_token is not None and Context.hf_token != api_key:
            warnings.warn(
                """You are loading a model/Space with a different access token than the one you used to load a previous model/Space. This is not recommended, as it may cause unexpected behavior."""
            )
        Context.hf_token = api_key

    blocks: gradio.Blocks = factory_methods[src](name, api_key, alias, **kwargs)
    return blocks


def chatbot_preprocess(text, state):
    payload = {
        "inputs": {"generated_responses": None, "past_user_inputs": None, "text": text}
    }
    if state is not None:
        payload["inputs"]["generated_responses"] = state["conversation"][
            "generated_responses"
        ]
        payload["inputs"]["past_user_inputs"] = state["conversation"][
            "past_user_inputs"
        ]

    return payload


def chatbot_postprocess(response):
    response_json = response.json()
    chatbot_value = list(
        zip(
            response_json["conversation"]["past_user_inputs"],
            response_json["conversation"]["generated_responses"],
        )
    )
    return chatbot_value, response_json


def from_model(model_name: str, api_key: str | None, alias: str | None, **kwargs):
    model_url = f"https://huggingface.co/{model_name}"
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    print(f"Fetching model from: {model_url}")

    headers = {"Authorization": f"Bearer {api_key}"} if api_key is not None else {}

    # Checking if model exists, and if so, it gets the pipeline
    response = requests.request("GET", api_url, headers=headers)
    assert (
        response.status_code == 200
    ), f"Could not find model: {model_name}. If it is a private or gated model, please provide your Hugging Face access token (https://huggingface.co/settings/tokens) as the argument for the `api_key` parameter."
    p = response.json().get("pipeline_tag")
    pipelines = {
        "audio-classification": {
            # example model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
            "inputs": components.Audio(source="upload", type="filepath", label="Input"),
            "outputs": components.Label(label="Class"),
            "preprocess": lambda i: to_binary,
            "postprocess": lambda r: postprocess_label(
                {i["label"].split(", ")[0]: i["score"] for i in r.json()}
            ),
        },
        "audio-to-audio": {
            # example model: facebook/xm_transformer_sm_all-en
            "inputs": components.Audio(source="upload", type="filepath", label="Input"),
            "outputs": components.Audio(label="Output"),
            "preprocess": to_binary,
            "postprocess": encode_to_base64,
        },
        "automatic-speech-recognition": {
            # example model: facebook/wav2vec2-base-960h
            "inputs": components.Audio(source="upload", type="filepath", label="Input"),
            "outputs": components.Textbox(label="Output"),
            "preprocess": to_binary,
            "postprocess": lambda r: r.json()["text"],
        },
        "conversational": {
            "inputs": [components.Textbox(), components.State()],  # type: ignore
            "outputs": [components.Chatbot(), components.State()],  # type: ignore
            "preprocess": chatbot_preprocess,
            "postprocess": chatbot_postprocess,
        },
        "feature-extraction": {
            # example model: julien-c/distilbert-feature-extraction
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Dataframe(label="Output"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: r.json()[0],
        },
        "fill-mask": {
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Label(label="Classification"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: postprocess_label(
                {i["token_str"]: i["score"] for i in r.json()}
            ),
        },
        "image-classification": {
            # Example: google/vit-base-patch16-224
            "inputs": components.Image(type="filepath", label="Input Image"),
            "outputs": components.Label(label="Classification"),
            "preprocess": to_binary,
            "postprocess": lambda r: postprocess_label(
                {i["label"].split(", ")[0]: i["score"] for i in r.json()}
            ),
        },
        "question-answering": {
            # Example: deepset/xlm-roberta-base-squad2
            "inputs": [
                components.Textbox(lines=7, label="Context"),
                components.Textbox(label="Question"),
            ],
            "outputs": [
                components.Textbox(label="Answer"),
                components.Label(label="Score"),
            ],
            "preprocess": lambda c, q: {"inputs": {"context": c, "question": q}},
            "postprocess": lambda r: (r.json()["answer"], {"label": r.json()["score"]}),
        },
        "summarization": {
            # Example: facebook/bart-large-cnn
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Textbox(label="Summary"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: r.json()[0]["summary_text"],
        },
        "text-classification": {
            # Example: distilbert-base-uncased-finetuned-sst-2-english
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Label(label="Classification"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: postprocess_label(
                {i["label"].split(", ")[0]: i["score"] for i in r.json()[0]}
            ),
        },
        "text-generation": {
            # Example: gpt2
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Textbox(label="Output"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: r.json()[0]["generated_text"],
        },
        "text2text-generation": {
            # Example: valhalla/t5-small-qa-qg-hl
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Textbox(label="Generated Text"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: r.json()[0]["generated_text"],
        },
        "translation": {
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Textbox(label="Translation"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: r.json()[0]["translation_text"],
        },
        "zero-shot-classification": {
            # Example: facebook/bart-large-mnli
            "inputs": [
                components.Textbox(label="Input"),
                components.Textbox(label="Possible class names (" "comma-separated)"),
                components.Checkbox(label="Allow multiple true classes"),
            ],
            "outputs": components.Label(label="Classification"),
            "preprocess": lambda i, c, m: {
                "inputs": i,
                "parameters": {"candidate_labels": c, "multi_class": m},
            },
            "postprocess": lambda r: postprocess_label(
                {
                    r.json()["labels"][i]: r.json()["scores"][i]
                    for i in range(len(r.json()["labels"]))
                }
            ),
        },
        "sentence-similarity": {
            # Example: sentence-transformers/distilbert-base-nli-stsb-mean-tokens
            "inputs": [
                components.Textbox(
                    value="That is a happy person", label="Source Sentence"
                ),
                components.Textbox(
                    lines=7,
                    placeholder="Separate each sentence by a newline",
                    label="Sentences to compare to",
                ),
            ],
            "outputs": components.Label(label="Classification"),
            "preprocess": lambda src, sentences: {
                "inputs": {
                    "source_sentence": src,
                    "sentences": [s for s in sentences.splitlines() if s != ""],
                }
            },
            "postprocess": lambda r: postprocess_label(
                {f"sentence {i}": v for i, v in enumerate(r.json())}
            ),
        },
        "text-to-speech": {
            # Example: julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Audio(label="Audio"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": encode_to_base64,
        },
        "text-to-image": {
            # example model: osanseviero/BigGAN-deep-128
            "inputs": components.Textbox(label="Input"),
            "outputs": components.Image(label="Output"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": encode_to_base64,
        },
        "token-classification": {
            # example model: huggingface-course/bert-finetuned-ner
            "inputs": components.Textbox(label="Input"),
            "outputs": components.HighlightedText(label="Output"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: r,  # Handled as a special case in query_huggingface_api()
        },
        "document-question-answering": {
            # example model: impira/layoutlm-document-qa
            "inputs": [
                components.Image(type="filepath", label="Input Document"),
                components.Textbox(label="Question"),
            ],
            "outputs": components.Label(label="Label"),
            "preprocess": lambda img, q: {
                "inputs": {
                    "image": extract_base64_data(img),  # Extract base64 data
                    "question": q,
                }
            },
            "postprocess": lambda r: postprocess_label(
                {i["answer"]: i["score"] for i in r.json()}
            ),
        },
        "visual-question-answering": {
            # example model: dandelin/vilt-b32-finetuned-vqa
            "inputs": [
                components.Image(type="filepath", label="Input Image"),
                components.Textbox(label="Question"),
            ],
            "outputs": components.Label(label="Label"),
            "preprocess": lambda img, q: {
                "inputs": {
                    "image": extract_base64_data(img),
                    "question": q,
                }
            },
            "postprocess": lambda r: postprocess_label(
                {i["answer"]: i["score"] for i in r.json()}
            ),
        },
        "image-to-text": {
            # example model: Salesforce/blip-image-captioning-base
            "inputs": components.Image(type="filepath", label="Input Image"),
            "outputs": components.Textbox(label="Generated Text"),
            "preprocess": to_binary,
            "postprocess": lambda r: r.json()[0]["generated_text"],
        },
    }

    if p in ["tabular-classification", "tabular-regression"]:
        example_data = get_tabular_examples(model_name)
        col_names, example_data = cols_to_rows(example_data)
        example_data = [[example_data]] if example_data else None

        pipelines[p] = {
            "inputs": components.Dataframe(
                label="Input Rows",
                type="pandas",
                headers=col_names,
                col_count=(len(col_names), "fixed"),
            ),
            "outputs": components.Dataframe(
                label="Predictions", type="array", headers=["prediction"]
            ),
            "preprocess": rows_to_cols,
            "postprocess": lambda r: {
                "headers": ["prediction"],
                "data": [[pred] for pred in json.loads(r.text)],
            },
            "examples": example_data,
        }

    if p is None or p not in pipelines:
        raise ValueError(f"Unsupported pipeline type: {p}")

    pipeline = pipelines[p]

    def query_huggingface_api(*params):
        # Convert to a list of input components
        data = pipeline["preprocess"](*params)
        if isinstance(
            data, dict
        ):  # HF doesn't allow additional parameters for binary files (e.g. images or audio files)
            data.update({"options": {"wait_for_model": True}})
            data = json.dumps(data)
        response = requests.request("POST", api_url, headers=headers, data=data)
        if response.status_code != 200:
            errors_json = response.json()
            errors, warns = "", ""
            if errors_json.get("error"):
                errors = f", Error: {errors_json.get('error')}"
            if errors_json.get("warnings"):
                warns = f", Warnings: {errors_json.get('warnings')}"
            raise Error(
                f"Could not complete request to HuggingFace API, Status Code: {response.status_code}"
                + errors
                + warns
            )
        if (
            p == "token-classification"
        ):  # Handle as a special case since HF API only returns the named entities and we need the input as well
            ner_groups = response.json()
            input_string = params[0]
            response = utils.format_ner_list(input_string, ner_groups)
        output = pipeline["postprocess"](response)
        return output

    if alias is None:
        query_huggingface_api.__name__ = model_name
    else:
        query_huggingface_api.__name__ = alias

    interface_info = {
        "fn": query_huggingface_api,
        "inputs": pipeline["inputs"],
        "outputs": pipeline["outputs"],
        "title": model_name,
        "examples": pipeline.get("examples"),
    }

    kwargs = dict(interface_info, **kwargs)

    # So interface doesn't run pre/postprocess
    # except for conversational interfaces which
    # are stateful
    kwargs["_api_mode"] = p != "conversational"

    interface = gradio.Interface(**kwargs)
    return interface


def from_spaces(
    space_name: str, api_key: str | None, alias: str | None, **kwargs
) -> Blocks:
    space_url = f"https://huggingface.co/spaces/{space_name}"

    print(f"Fetching Space from: {space_url}")

    headers = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    iframe_url = (
        requests.get(
            f"https://huggingface.co/api/spaces/{space_name}/host", headers=headers
        )
        .json()
        .get("host")
    )

    if iframe_url is None:
        raise ValueError(
            f"Could not find Space: {space_name}. If it is a private or gated Space, please provide your Hugging Face access token (https://huggingface.co/settings/tokens) as the argument for the `api_key` parameter."
        )

    r = requests.get(iframe_url, headers=headers)

    result = re.search(
        r"window.gradio_config = (.*?);[\s]*</script>", r.text
    )  # some basic regex to extract the config
    try:
        config = json.loads(result.group(1))  # type: ignore
    except AttributeError as ae:
        raise ValueError(f"Could not load the Space: {space_name}") from ae
    if "allow_flagging" in config:  # Create an Interface for Gradio 2.x Spaces
        return from_spaces_interface(
            space_name, config, alias, api_key, iframe_url, **kwargs
        )
    else:  # Create a Blocks for Gradio 3.x Spaces
        if kwargs:
            warnings.warn(
                "You cannot override parameters for this Space by passing in kwargs. "
                "Instead, please load the Space as a function and use it to create a "
                "Blocks or Interface locally. You may find this Guide helpful: "
                "https://gradio.app/using_blocks_like_functions/"
            )
        return from_spaces_blocks(space=space_name, api_key=api_key)


def from_spaces_blocks(space: str, api_key: str | None) -> Blocks:
    client = Client(space, hf_token=api_key)
    predict_fns = [endpoint._predict_resolve for endpoint in client.endpoints]
    return gradio.Blocks.from_config(client.config, predict_fns, client.src)


def from_spaces_interface(
    model_name: str,
    config: dict,
    alias: str | None,
    api_key: str | None,
    iframe_url: str,
    **kwargs,
) -> Interface:
    config = streamline_spaces_interface(config)
    api_url = f"{iframe_url}/api/predict/"
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    # The function should call the API with preprocessed data
    def fn(*data):
        data = json.dumps({"data": data})
        response = requests.post(api_url, headers=headers, data=data)
        result = json.loads(response.content.decode("utf-8"))
        if "error" in result and "429" in result["error"]:
            raise TooManyRequestsError("Too many requests to the Hugging Face API")
        try:
            output = result["data"]
        except KeyError as ke:
            raise KeyError(
                f"Could not find 'data' key in response from external Space. Response received: {result}"
            ) from ke
        if (
            len(config["outputs"]) == 1
        ):  # if the fn is supposed to return a single value, pop it
            output = output[0]
        if len(config["outputs"]) == 1 and isinstance(
            output, list
        ):  # Needed to support Output.Image() returning bounding boxes as well (TODO: handle different versions of gradio since they have slightly different APIs)
            output = output[0]
        return output

    fn.__name__ = alias if (alias is not None) else model_name
    config["fn"] = fn

    kwargs = dict(config, **kwargs)
    kwargs["_api_mode"] = True
    interface = gradio.Interface(**kwargs)
    return interface
