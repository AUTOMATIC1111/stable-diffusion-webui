from __future__ import annotations

import csv
import datetime
import json
import os
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from distutils.version import StrictVersion
from pathlib import Path
from typing import TYPE_CHECKING, Any

import filelock
import huggingface_hub
import pkg_resources
from gradio_client import utils as client_utils
from gradio_client.documentation import document, set_documentation_group

import gradio as gr
from gradio import utils

if TYPE_CHECKING:
    from gradio.components import IOComponent

set_documentation_group("flagging")


class FlaggingCallback(ABC):
    """
    An abstract class for defining the methods that any FlaggingCallback should have.
    """

    @abstractmethod
    def setup(self, components: list[IOComponent], flagging_dir: str):
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
        flag_data: list[Any],
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

    def setup(self, components: list[IOComponent], flagging_dir: str | Path):
        self.components = components
        self.flagging_dir = flagging_dir
        os.makedirs(flagging_dir, exist_ok=True)

    def flag(
        self,
        flag_data: list[Any],
        flag_option: str = "",
        username: str | None = None,
    ) -> int:
        flagging_dir = self.flagging_dir
        log_filepath = Path(flagging_dir) / "log.csv"

        csv_data = []
        for component, sample in zip(self.components, flag_data):
            save_dir = Path(
                flagging_dir
            ) / client_utils.strip_invalid_filename_characters(component.label or "")
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

        with open(log_filepath) as csvfile:
            line_count = len(list(csv.reader(csvfile))) - 1
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
    Guides: using-flagging
    """

    def __init__(self):
        pass

    def setup(
        self,
        components: list[IOComponent],
        flagging_dir: str | Path,
    ):
        self.components = components
        self.flagging_dir = flagging_dir
        os.makedirs(flagging_dir, exist_ok=True)

    def flag(
        self,
        flag_data: list[Any],
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
            save_dir = Path(
                flagging_dir
            ) / client_utils.strip_invalid_filename_characters(
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

        with open(log_filepath, encoding="utf-8") as csvfile:
            line_count = len(list(csv.reader(csvfile))) - 1
        return line_count


@document()
class HuggingFaceDatasetSaver(FlaggingCallback):
    """
    A callback that saves each flagged sample (both the input and output data) to a HuggingFace dataset.

    Example:
        import gradio as gr
        hf_writer = gr.HuggingFaceDatasetSaver(HF_API_TOKEN, "image-classification-mistakes")
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            allow_flagging="manual", flagging_callback=hf_writer)
    Guides: using-flagging
    """

    def __init__(
        self,
        hf_token: str,
        dataset_name: str,
        organization: str | None = None,
        private: bool = False,
        info_filename: str = "dataset_info.json",
        separate_dirs: bool = False,
        verbose: bool = True,  # silently ignored. TODO: remove it?
    ):
        """
        Parameters:
            hf_token: The HuggingFace token to use to create (and write the flagged sample to) the HuggingFace dataset (defaults to the registered one).
            dataset_name: The repo_id of the dataset to save the data to, e.g. "image-classifier-1" or "username/image-classifier-1".
            organization: Deprecated argument. Please pass a full dataset id (e.g. 'username/dataset_name') to `dataset_name` instead.
            private: Whether the dataset should be private (defaults to False).
            info_filename: The name of the file to save the dataset info (defaults to "dataset_infos.json").
            separate_dirs: If True, each flagged item will be saved in a separate directory. This makes the flagging more robust to concurrent editing, but may be less convenient to use.
        """
        if organization is not None:
            warnings.warn(
                "Parameter `organization` is not used anymore. Please pass a full dataset id (e.g. 'username/dataset_name') to `dataset_name` instead."
            )
        self.hf_token = hf_token
        self.dataset_id = dataset_name  # TODO: rename parameter (but ensure backward compatibility somehow)
        self.dataset_private = private
        self.info_filename = info_filename
        self.separate_dirs = separate_dirs

    def setup(self, components: list[IOComponent], flagging_dir: str):
        """
        Params:
        flagging_dir (str): local directory where the dataset is cloned,
        updated, and pushed from.
        """
        hh_version = pkg_resources.get_distribution("huggingface_hub").version
        try:
            if StrictVersion(hh_version) < StrictVersion("0.12.0"):
                raise ImportError(
                    "The `huggingface_hub` package must be version 0.12.0 or higher"
                    "for HuggingFaceDatasetSaver. Try 'pip install huggingface_hub --upgrade'."
                )
        except ValueError:
            pass

        # Setup dataset on the Hub
        self.dataset_id = huggingface_hub.create_repo(
            repo_id=self.dataset_id,
            token=self.hf_token,
            private=self.dataset_private,
            repo_type="dataset",
            exist_ok=True,
        ).repo_id

        # Setup flagging dir
        self.components = components
        self.dataset_dir = (
            Path(flagging_dir).absolute() / self.dataset_id.split("/")[-1]
        )
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.infos_file = self.dataset_dir / self.info_filename

        # Download remote files to local
        remote_files = [self.info_filename]
        if not self.separate_dirs:
            # No separate dirs => means all data is in the same CSV file => download it to get its current content
            remote_files.append("data.csv")

        for filename in remote_files:
            try:
                huggingface_hub.hf_hub_download(
                    repo_id=self.dataset_id,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=self.dataset_dir,
                    token=self.hf_token,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                pass

    def flag(
        self,
        flag_data: list[Any],
        flag_option: str = "",
        username: str | None = None,
    ) -> int:
        if self.separate_dirs:
            # JSONL files to support dataset preview on the Hub
            unique_id = str(uuid.uuid4())
            components_dir = self.dataset_dir / str(uuid.uuid4())
            data_file = components_dir / "metadata.jsonl"
            path_in_repo = unique_id  # upload in sub folder (safer for concurrency)
        else:
            # Unique CSV file
            components_dir = self.dataset_dir
            data_file = components_dir / "data.csv"
            path_in_repo = None  # upload at root level

        return self._flag_in_dir(
            data_file=data_file,
            components_dir=components_dir,
            path_in_repo=path_in_repo,
            flag_data=flag_data,
            flag_option=flag_option,
            username=username or "",
        )

    def _flag_in_dir(
        self,
        data_file: Path,
        components_dir: Path,
        path_in_repo: str | None,
        flag_data: list[Any],
        flag_option: str = "",
        username: str = "",
    ) -> int:
        # Deserialize components (write images/audio to files)
        features, row = self._deserialize_components(
            components_dir, flag_data, flag_option, username
        )

        # Write generic info to dataset_infos.json + upload
        with filelock.FileLock(str(self.infos_file) + ".lock"):
            if not self.infos_file.exists():
                self.infos_file.write_text(
                    json.dumps({"flagged": {"features": features}})
                )

                huggingface_hub.upload_file(
                    repo_id=self.dataset_id,
                    repo_type="dataset",
                    token=self.hf_token,
                    path_in_repo=self.infos_file.name,
                    path_or_fileobj=self.infos_file,
                )

        headers = list(features.keys())

        if not self.separate_dirs:
            with filelock.FileLock(components_dir / ".lock"):
                sample_nb = self._save_as_csv(data_file, headers=headers, row=row)
                sample_name = str(sample_nb)
                huggingface_hub.upload_folder(
                    repo_id=self.dataset_id,
                    repo_type="dataset",
                    commit_message=f"Flagged sample #{sample_name}",
                    path_in_repo=path_in_repo,
                    ignore_patterns="*.lock",
                    folder_path=components_dir,
                    token=self.hf_token,
                )
        else:
            sample_name = self._save_as_jsonl(data_file, headers=headers, row=row)
            sample_nb = len(
                [path for path in self.dataset_dir.iterdir() if path.is_dir()]
            )
            huggingface_hub.upload_folder(
                repo_id=self.dataset_id,
                repo_type="dataset",
                commit_message=f"Flagged sample #{sample_name}",
                path_in_repo=path_in_repo,
                ignore_patterns="*.lock",
                folder_path=components_dir,
                token=self.hf_token,
            )

        return sample_nb

    @staticmethod
    def _save_as_csv(data_file: Path, headers: list[str], row: list[Any]) -> int:
        """Save data as CSV and return the sample name (row number)."""
        is_new = not data_file.exists()

        with data_file.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write CSV headers if new file
            if is_new:
                writer.writerow(utils.sanitize_list_for_csv(headers))

            # Write CSV row for flagged sample
            writer.writerow(utils.sanitize_list_for_csv(row))

        with data_file.open(encoding="utf-8") as csvfile:
            return sum(1 for _ in csv.reader(csvfile)) - 1

    @staticmethod
    def _save_as_jsonl(data_file: Path, headers: list[str], row: list[Any]) -> str:
        """Save data as JSONL and return the sample name (uuid)."""
        Path.mkdir(data_file.parent, parents=True, exist_ok=True)
        with open(data_file, "w") as f:
            json.dump(dict(zip(headers, row)), f)
        return data_file.parent.name

    def _deserialize_components(
        self,
        data_dir: Path,
        flag_data: list[Any],
        flag_option: str = "",
        username: str = "",
    ) -> tuple[dict[Any, Any], list[Any]]:
        """Deserialize components and return the corresponding row for the flagged sample.

        Images/audio are saved to disk as individual files.
        """
        # Components that can have a preview on dataset repos
        file_preview_types = {gr.Audio: "Audio", gr.Image: "Image"}

        # Generate the row corresponding to the flagged sample
        features = OrderedDict()
        row = []
        for component, sample in zip(self.components, flag_data):
            # Get deserialized object (will save sample to disk if applicable -file, audio, image,...-)
            label = component.label or ""
            save_dir = data_dir / client_utils.strip_invalid_filename_characters(label)
            deserialized = component.deserialize(sample, save_dir, None)

            # Add deserialized object to row
            features[label] = {"dtype": "string", "_type": "Value"}
            try:
                assert Path(deserialized).exists()
                row.append(Path(deserialized).name)
            except (AssertionError, TypeError, ValueError):
                row.append(str(deserialized))

            # If component is eligible for a preview, add the URL of the file
            if isinstance(component, tuple(file_preview_types)):  # type: ignore
                for _component, _type in file_preview_types.items():
                    if isinstance(component, _component):
                        features[label + " file"] = {"_type": _type}
                        break
                path_in_repo = str(  # returned filepath is absolute, we want it relative to compute URL
                    Path(deserialized).relative_to(self.dataset_dir)
                ).replace(
                    "\\", "/"
                )
                row.append(
                    huggingface_hub.hf_hub_url(
                        repo_id=self.dataset_id,
                        filename=path_in_repo,
                        repo_type="dataset",
                    )
                )
        features["flag"] = {"dtype": "string", "_type": "Value"}
        features["username"] = {"dtype": "string", "_type": "Value"}
        row.append(flag_option)
        row.append(username)
        return features, row


class HuggingFaceDatasetJSONSaver(HuggingFaceDatasetSaver):
    def __init__(
        self,
        hf_token: str,
        dataset_name: str,
        organization: str | None = None,
        private: bool = False,
        info_filename: str = "dataset_info.json",
        verbose: bool = True,  # silently ignored. TODO: remove it?
    ):
        warnings.warn(
            "Callback `HuggingFaceDatasetJSONSaver` is deprecated in favor of using"
            " `HuggingFaceDatasetSaver` and passing `separate_dirs=True` as parameter."
        )
        super().__init__(
            hf_token=hf_token,
            dataset_name=dataset_name,
            organization=organization,
            private=private,
            info_filename=info_filename,
            separate_dirs=True,
        )


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

    def __call__(self, request: gr.Request, *flag_data):
        try:
            self.flagging_callback.flag(
                list(flag_data), flag_option=self.value, username=request.username
            )
        except Exception as e:
            print(f"Error while flagging: {e}")
            if self.visual_feedback:
                return "Error!"
        if not self.visual_feedback:
            return
        time.sleep(0.8)  # to provide enough time for the user to observe button change
        return self.reset()

    def reset(self):
        return gr.Button.update(value=self.label, interactive=True)
