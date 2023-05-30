""" Functions related to analytics and telemetry. """
from __future__ import annotations

import json
import os
import pkgutil
import threading
import warnings
from distutils.version import StrictVersion
from typing import Any

import requests

import gradio
from gradio.context import Context
from gradio.utils import GRADIO_VERSION

ANALYTICS_URL = "https://api.gradio.app/"
PKG_VERSION_URL = "https://api.gradio.app/pkg-version"


def analytics_enabled() -> bool:
    """
    Returns: True if analytics are enabled, False otherwise.
    """
    return os.getenv("GRADIO_ANALYTICS_ENABLED", "True") == "True"


def _do_analytics_request(url: str, data: dict[str, Any]) -> None:
    try:
        requests.post(url, data=data, timeout=5)
    except (requests.ConnectionError, requests.exceptions.ReadTimeout):
        pass  # do not push analytics if no network


def version_check():
    if not analytics_enabled():
        return
    try:
        version_data = pkgutil.get_data(__name__, "version.txt")
        if not version_data:
            raise FileNotFoundError
        current_pkg_version = version_data.decode("ascii").strip()
        latest_pkg_version = requests.get(url=PKG_VERSION_URL, timeout=3).json()[
            "version"
        ]
        if StrictVersion(latest_pkg_version) > StrictVersion(current_pkg_version):
            print(
                f"IMPORTANT: You are using gradio version {current_pkg_version}, "
                f"however version {latest_pkg_version} is available, please upgrade."
            )
            print("--------")
    except json.decoder.JSONDecodeError:
        warnings.warn("unable to parse version details from package URL.")
    except KeyError:
        warnings.warn("package URL does not contain version info.")
    except Exception:
        pass


def get_local_ip_address() -> str:
    """
    Gets the public IP address or returns the string "No internet connection" if unable
    to obtain it or the string "Analytics disabled" if a user has disabled analytics.
    Does not make a new request if the IP address has already been obtained in the
    same Python session.
    """
    if not analytics_enabled():
        return "Analytics disabled"

    if Context.ip_address is None:
        try:
            ip_address = requests.get(
                "https://checkip.amazonaws.com/", timeout=3
            ).text.strip()
        except (requests.ConnectionError, requests.exceptions.ReadTimeout):
            ip_address = "No internet connection"
        Context.ip_address = ip_address
    else:
        ip_address = Context.ip_address
    return ip_address


def initiated_analytics(data: dict[str, Any]) -> None:
    if not analytics_enabled():
        return

    threading.Thread(
        target=_do_analytics_request,
        kwargs={
            "url": f"{ANALYTICS_URL}gradio-initiated-analytics/",
            "data": {**data, "ip_address": get_local_ip_address()},
        },
    ).start()


def launched_analytics(blocks: gradio.Blocks, data: dict[str, Any]) -> None:
    if not analytics_enabled():
        return

    blocks_telemetry, inputs_telemetry, outputs_telemetry, targets_telemetry = (
        [],
        [],
        [],
        [],
    )

    from gradio.blocks import BlockContext

    for x in list(blocks.blocks.values()):
        blocks_telemetry.append(x.get_block_name()) if isinstance(
            x, BlockContext
        ) else blocks_telemetry.append(str(x))

    for x in blocks.dependencies:
        targets_telemetry = targets_telemetry + [
            str(blocks.blocks[y]) for y in x["targets"]
        ]
        inputs_telemetry = inputs_telemetry + [
            str(blocks.blocks[y]) for y in x["inputs"]
        ]
        outputs_telemetry = outputs_telemetry + [
            str(blocks.blocks[y]) for y in x["outputs"]
        ]
    additional_data = {
        "version": GRADIO_VERSION,
        "is_kaggle": blocks.is_kaggle,
        "is_sagemaker": blocks.is_sagemaker,
        "using_auth": blocks.auth is not None,
        "dev_mode": blocks.dev_mode,
        "show_api": blocks.show_api,
        "show_error": blocks.show_error,
        "title": blocks.title,
        "inputs": blocks.input_components
        if blocks.mode == "interface"
        else inputs_telemetry,
        "outputs": blocks.output_components
        if blocks.mode == "interface"
        else outputs_telemetry,
        "targets": targets_telemetry,
        "blocks": blocks_telemetry,
        "events": [str(x["trigger"]) for x in blocks.dependencies],
    }

    data.update(additional_data)
    data.update({"ip_address": get_local_ip_address()})

    threading.Thread(
        target=_do_analytics_request,
        kwargs={
            "url": f"{ANALYTICS_URL}gradio-launched-telemetry/",
            "data": data,
        },
    ).start()


def integration_analytics(data: dict[str, Any]) -> None:
    if not analytics_enabled():
        return

    threading.Thread(
        target=_do_analytics_request,
        kwargs={
            "url": f"{ANALYTICS_URL}gradio-integration-analytics/",
            "data": {**data, "ip_address": get_local_ip_address()},
        },
    ).start()


def error_analytics(message: str) -> None:
    """
    Send error analytics if there is network
    Parameters:
        message: Details about error
    """
    if not analytics_enabled():
        return

    data = {"ip_address": get_local_ip_address(), "error": message}

    threading.Thread(
        target=_do_analytics_request,
        kwargs={
            "url": f"{ANALYTICS_URL}gradio-error-analytics/",
            "data": data,
        },
    ).start()
