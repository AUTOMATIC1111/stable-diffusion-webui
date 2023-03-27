from __future__ import annotations

import csv
import datetime
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from distutils.version import StrictVersion
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import pkg_resources

import gradio as gr
from gradio import utils
from gradio.documentation import document, set_documentation_group

if TYPE_CHECKING:
    from gradio.components import IOComponent

set_documentation_group("flagging")


def _get_dataset_features_info(is_new, components):
    """
    Takes in a list of components and returns a dataset features info

    Parameters:
    is_new: boolean, whether the dataset is new or not
    components: list of components

    Returns:
    infos: a dictionary of the dataset features
    file_preview_types: dictionary mapping of gradio components to appropriate string.
    header: list of header strings

    """
    infos = {"flagged": {"features": {}}}
    # File previews for certain input and output types
    file_preview_types = {gr.Audio: "Audio", gr.Image: "Image"}
    headers = []

    # Generate the headers and dataset_infos
    if is_new:

        for component in components:
            headers.append(component.label)
            infos["flagged"]["features"][component.label] = {
                "dtype": "string",
                "_type": "Value",
            }
            if isinstance(component, tuple(file_preview_types)):
                headers.append(component.label + " file")
                for _component, _type in file_preview_types.items():
                    if isinstance(component, _component):
                        infos["flagged"]["features"][
                            (component.label or "") + " file"
                        ] = {"_type": _type}
                        break

        headers.append("flag")
        infos["flagged"]["features"]["flag"] = {
            "dtype": "string",
            "_type": "Value",
        }

    return infos, file_preview_types, headers


class FlaggingCallback(ABC):
    """
    An abstract class for defining the methods that any FlaggingCallback should have.
    """

    @abstractmethod
    def setup(self, components: List[IOComponent], flagging_dir: str):
        """
        This method should be overridden and ensure that everything is set up correctly for flag().
        This method gets called once at the beginning of the Interface.launch() method.
        Parameters:
        components: Set of components that will provide flagged data.
        flagging_dir: A string, typically containing the path to the directory where the flagging file should be storied (provided as an argument to Interface.__init__()).
        """
        pass

    @abstractmethod
    def flag(
        self,
        flag_data: List[Any],
        flag_option: str = "",
        username: str | None = None,
    ) -> int:
        """
        This method should be overridden by the FlaggingCallback subclass and may contain optional additional arguments.
        This gets called every time the <flag> button is pressed.
        Parameters:
        interface: The Interface object that is being used to launch the flagging interface.
        flag_data: The data to be flagged.
        flag_option (optional): In the case that flagging_options are provided, the flag option that is being used.
        username (optional): The username of the user that is flagging the data, if logged in.
        Returns:
        (int) The total number of samples that have been flagged.
        """
        pass


@document()
class SimpleCSVLogger(FlaggingCallback):
    """
    A simplified implementation of the FlaggingCallback abstract class
    provided for illustrative purposes.  Each flagged sample (both the input and output data)
    is logged to a CSV file on the machine running the gradio app.
    Example:
        import gradio as gr
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            flagging_callback=SimpleCSVLogger())
    """

    def __init__(self):
        pass

    def setup(self, components: List[IOComponent], flagging_dir: str | Path):
        self.components = components
        self.flagging_dir = flagging_dir
        os.makedirs(flagging_dir, exist_ok=True)

    def flag(
        self,
        flag_data: List[Any],
        flag_option: str = "",
        username: str | None = None,
    ) -> int:
        flagging_dir = self.flagging_dir
        log_filepath = Path(flagging_dir) / "log.csv"

        csv_data = []
        for component, sample in zip(self.components, flag_data):
            save_dir = Path(flagging_dir) / utils.strip_invalid_filename_characters(
                component.label or ""
            )
            csv_data.append(
                component.deserialize(
                    sample,
                    save_dir,
                    None,
                )
            )

        with open(log_filepath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(utils.sanitize_list_for_csv(csv_data))

        with open(log_filepath, "r") as csvfile:
            line_count = len([None for row in csv.reader(csvfile)]) - 1
        return line_count


@document()
class CSVLogger(FlaggingCallback):
    """
    The default implementation of the FlaggingCallback abstract class. Each flagged
    sample (both the input and output data) is logged to a CSV file with headers on the machine running the gradio app.
    Example:
        import gradio as gr
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            flagging_callback=CSVLogger())
    Guides: using_flagging
    """

    def __init__(self):
        pass

    def setup(
        self,
        components: List[IOComponent],
        flagging_dir: str | Path,
    ):
        self.components = components
        self.flagging_dir = flagging_dir
        os.makedirs(flagging_dir, exist_ok=True)

    def flag(
        self,
        flag_data: List[Any],
        flag_option: str = "",
        username: str | None = None,
    ) -> int:
        flagging_dir = self.flagging_dir
        log_filepath = Path(flagging_dir) / "log.csv"
        is_new = not Path(log_filepath).exists()
        headers = [
            getattr(component, "label", None) or f"component {idx}"
            for idx, component in enumerate(self.components)
        ] + [
            "flag",
            "username",
            "timestamp",
        ]

        csv_data = []
        for idx, (component, sample) in enumerate(zip(self.components, flag_data)):
            save_dir = Path(flagging_dir) / utils.strip_invalid_filename_characters(
                getattr(component, "label", None) or f"component {idx}"
            )
            if utils.is_update(sample):
                csv_data.append(str(sample))
            else:
                csv_data.append(
                    component.deserialize(sample, save_dir=save_dir)
                    if sample is not None
                    else ""
                )
        csv_data.append(flag_option)
        csv_data.append(username if username is not None else "")
        csv_data.append(str(datetime.datetime.now()))

        with open(log_filepath, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if is_new:
                writer.writerow(utils.sanitize_list_for_csv(headers))
            writer.writerow(utils.sanitize_list_for_csv(csv_data))

        with open(log_filepath, "r", encoding="utf-8") as csvfile:
            line_count = len([None for row in csv.reader(csvfile)]) - 1
        return line_count


@document()
class HuggingFaceDatasetSaver(FlaggingCallback):
    """
    A callback that saves each flagged sample (both the input and output data)
    to a HuggingFace dataset.
    Example:
        import gradio as gr
        hf_writer = gr.HuggingFaceDatasetSaver(HF_API_TOKEN, "image-classification-mistakes")
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            allow_flagging="manual", flagging_callback=hf_writer)
    Guides: using_flagging
    """

    def __init__(
        self,
        hf_token: str,
        dataset_name: str,
        organization: str | None = None,
        private: bool = False,
    ):
        """
        Parameters:
            hf_token: The HuggingFace token to use to create (and write the flagged sample to) the HuggingFace dataset.
            dataset_name: The name of the dataset to save the data to, e.g. "image-classifier-1"
            organization: The organization to save the dataset under. The hf_token must provide write access to this organization. If not provided, saved under the name of the user corresponding to the hf_token.
            private: Whether the dataset should be private (defaults to False).
        """
        self.hf_token = hf_token
        self.dataset_name = dataset_name
        self.organization_name = organization
        self.dataset_private = private

    def setup(self, components: List[IOComponent], flagging_dir: str):
        """
        Params:
        flagging_dir (str): local directory where the dataset is cloned,
        updated, and pushed from.
        """
        try:
            import huggingface_hub
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Package `huggingface_hub` not found is needed "
                "for HuggingFaceDatasetSaver. Try 'pip install huggingface_hub'."
            )
        hh_version = pkg_resources.get_distribution("huggingface_hub").version
        try:
            if StrictVersion(hh_version) < StrictVersion("0.6.0"):
                raise ImportError(
                    "The `huggingface_hub` package must be version 0.6.0 or higher"
                    "for HuggingFaceDatasetSaver. Try 'pip install huggingface_hub --upgrade'."
                )
        except ValueError:
            pass
        repo_id = huggingface_hub.get_full_repo_name(
            self.dataset_name, token=self.hf_token
        )
        path_to_dataset_repo = huggingface_hub.create_repo(
            repo_id=repo_id,
            token=self.hf_token,
            private=self.dataset_private,
            repo_type="dataset",
            exist_ok=True,
        )
        self.path_to_dataset_repo = path_to_dataset_repo  # e.g. "https://huggingface.co/datasets/abidlabs/test-audio-10"
        self.components = components
        self.flagging_dir = flagging_dir
        self.dataset_dir = Path(flagging_dir) / self.dataset_name
        self.repo = huggingface_hub.Repository(
            local_dir=str(self.dataset_dir),
            clone_from=path_to_dataset_repo,
            use_auth_token=self.hf_token,
        )
        self.repo.git_pull(lfs=True)

        # Should filename be user-specified?
        self.log_file = Path(self.dataset_dir) / "data.csv"
        self.infos_file = Path(self.dataset_dir) / "dataset_infos.json"

    def flag(
        self,
        flag_data: List[Any],
        flag_option: str = "",
        username: str | None = None,
    ) -> int:
        self.repo.git_pull(lfs=True)

        is_new = not Path(self.log_file).exists()

        with open(self.log_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # File previews for certain input and output types
            infos, file_preview_types, headers = _get_dataset_features_info(
                is_new, self.components
            )

            # Generate the headers and dataset_infos
            if is_new:
                writer.writerow(utils.sanitize_list_for_csv(headers))

            # Generate the row corresponding to the flagged sample
            csv_data = []
            for component, sample in zip(self.components, flag_data):
                save_dir = Path(
                    self.dataset_dir
                ) / utils.strip_invalid_filename_characters(component.label or "")
                filepath = component.deserialize(sample, save_dir, None)
                csv_data.append(filepath)
                if isinstance(component, tuple(file_preview_types)):
                    csv_data.append(
                        "{}/resolve/main/{}".format(self.path_to_dataset_repo, filepath)
                    )
            csv_data.append(flag_option)
            writer.writerow(utils.sanitize_list_for_csv(csv_data))

        if is_new:
            json.dump(infos, open(self.infos_file, "w"))

        with open(self.log_file, "r", encoding="utf-8") as csvfile:
            line_count = len([None for row in csv.reader(csvfile)]) - 1

        self.repo.push_to_hub(commit_message="Flagged sample #{}".format(line_count))

        return line_count


class HuggingFaceDatasetJSONSaver(FlaggingCallback):
    """
    A FlaggingCallback that saves flagged data to a Hugging Face dataset in JSONL format.

    Each data sample is saved in a different JSONL file,
    allowing multiple users to use flagging simultaneously.
    Saving to a single CSV would cause errors as only one user can edit at the same time.

    """

    def __init__(
        self,
        hf_token: str,
        dataset_name: str,
        organization: str | None = None,
        private: bool = False,
        verbose: bool = True,
    ):
        """
        Params:
        hf_token (str): The token to use to access the huggingface API.
        dataset_name (str): The name of the dataset to save the data to, e.g.
            "image-classifier-1"
        organization (str): The name of the organization to which to attach
            the datasets. If None, the dataset attaches to the user only.
        private (bool): If the dataset does not already exist, whether it
            should be created as a private dataset or public. Private datasets
            may require paid huggingface.co accounts
        verbose (bool): Whether to print out the status of the dataset
            creation.
        """
        self.hf_token = hf_token
        self.dataset_name = dataset_name
        self.organization_name = organization
        self.dataset_private = private
        self.verbose = verbose

    def setup(self, components: List[IOComponent], flagging_dir: str):
        """
        Params:
        components List[Component]: list of components for flagging
        flagging_dir (str): local directory where the dataset is cloned,
        updated, and pushed from.
        """
        try:
            import huggingface_hub
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Package `huggingface_hub` not found is needed "
                "for HuggingFaceDatasetJSONSaver. Try 'pip install huggingface_hub'."
            )
        hh_version = pkg_resources.get_distribution("huggingface_hub").version
        try:
            if StrictVersion(hh_version) < StrictVersion("0.6.0"):
                raise ImportError(
                    "The `huggingface_hub` package must be version 0.6.0 or higher"
                    "for HuggingFaceDatasetSaver. Try 'pip install huggingface_hub --upgrade'."
                )
        except ValueError:
            pass
        repo_id = huggingface_hub.get_full_repo_name(
            self.dataset_name, token=self.hf_token
        )
        path_to_dataset_repo = huggingface_hub.create_repo(
            repo_id=repo_id,
            token=self.hf_token,
            private=self.dataset_private,
            repo_type="dataset",
            exist_ok=True,
        )
        self.path_to_dataset_repo = path_to_dataset_repo  # e.g. "https://huggingface.co/datasets/abidlabs/test-audio-10"
        self.components = components
        self.flagging_dir = flagging_dir
        self.dataset_dir = Path(flagging_dir) / self.dataset_name
        self.repo = huggingface_hub.Repository(
            local_dir=str(self.dataset_dir),
            clone_from=path_to_dataset_repo,
            use_auth_token=self.hf_token,
        )
        self.repo.git_pull(lfs=True)

        self.infos_file = Path(self.dataset_dir) / "dataset_infos.json"

    def flag(
        self,
        flag_data: List[Any],
        flag_option: str = "",
        username: str | None = None,
    ) -> str:
        self.repo.git_pull(lfs=True)

        # Generate unique folder for the flagged sample
        unique_name = self.get_unique_name()  # unique name for folder
        folder_name = (
            Path(self.dataset_dir) / unique_name
        )  # unique folder for specific example
        os.makedirs(folder_name)

        # Now uses the existence of `dataset_infos.json` to determine if new
        is_new = not Path(self.infos_file).exists()

        # File previews for certain input and output types
        infos, file_preview_types, _ = _get_dataset_features_info(
            is_new, self.components
        )

        # Generate the row and header corresponding to the flagged sample
        csv_data = []
        headers = []

        for component, sample in zip(self.components, flag_data):
            headers.append(component.label)

            try:
                save_dir = Path(folder_name) / utils.strip_invalid_filename_characters(
                    component.label or ""
                )
                filepath = component.deserialize(sample, save_dir, None)
            except Exception:
                # Could not parse 'sample' (mostly) because it was None and `component.save_flagged`
                # does not handle None cases.
                # for example: Label (line 3109 of components.py raises an error if data is None)
                filepath = None

            if isinstance(component, tuple(file_preview_types)):
                headers.append(component.label or "" + " file")

                csv_data.append(
                    "{}/resolve/main/{}/{}".format(
                        self.path_to_dataset_repo, unique_name, filepath
                    )
                    if filepath is not None
                    else None
                )

            csv_data.append(filepath)
        headers.append("flag")
        csv_data.append(flag_option)

        # Creates metadata dict from row data and dumps it
        metadata_dict = {
            header: _csv_data for header, _csv_data in zip(headers, csv_data)
        }
        self.dump_json(metadata_dict, Path(folder_name) / "metadata.jsonl")

        if is_new:
            json.dump(infos, open(self.infos_file, "w"))

        self.repo.push_to_hub(commit_message="Flagged sample {}".format(unique_name))
        return unique_name

    def get_unique_name(self):
        id = uuid.uuid4()
        return str(id)

    def dump_json(self, thing: dict, file_path: str | Path) -> None:
        with open(file_path, "w+", encoding="utf8") as f:
            json.dump(thing, f)


class FlagMethod:
    """
    Helper class that contains the flagging options and calls the flagging method. Also
    provides visual feedback to the user when flag is clicked.
    """

    def __init__(
        self,
        flagging_callback: FlaggingCallback,
        label: str,
        value: str,
        visual_feedback: bool = True,
    ):
        self.flagging_callback = flagging_callback
        self.label = label
        self.value = value
        self.__name__ = "Flag"
        self.visual_feedback = visual_feedback

    def __call__(self, *flag_data):
        try:
            self.flagging_callback.flag(list(flag_data), flag_option=self.value)
        except Exception as e:
            print("Error while flagging: {}".format(e))
            if self.visual_feedback:
                return "Error!"
        if not self.visual_feedback:
            return
        time.sleep(0.8)  # to provide enough time for the user to observe button change
        return self.reset()

    def reset(self):
        return gr.Button.update(value=self.label, interactive=True)
