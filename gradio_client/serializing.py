from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from gradio_client import media_data, utils
from gradio_client.data_classes import FileData

with open(Path(__file__).parent / "types.json") as f:
    serializer_types = json.load(f)


class Serializable:
    def serialized_info(self):
        """
        The typing information for this component as a dictionary whose values are a list of 2 strings: [Python type, language-agnostic description].
        Keys of the dictionary are: raw_input, raw_output, serialized_input, serialized_output
        """
        return self.api_info()

    def api_info(self) -> dict[str, list[str]]:
        """
        The typing information for this component as a dictionary whose values are a list of 2 strings: [Python type, language-agnostic description].
        Keys of the dictionary are: raw_input, raw_output, serialized_input, serialized_output
        """
        raise NotImplementedError()

    def example_inputs(self) -> dict[str, Any]:
        """
        The example inputs for this component as a dictionary whose values are example inputs compatible with this component.
        Keys of the dictionary are: raw, serialized
        """
        raise NotImplementedError()

    # For backwards compatibility
    def input_api_info(self) -> tuple[str, str]:
        api_info = self.api_info()
        return (api_info["serialized_input"][0], api_info["serialized_input"][1])

    # For backwards compatibility
    def output_api_info(self) -> tuple[str, str]:
        api_info = self.api_info()
        return (api_info["serialized_output"][0], api_info["serialized_output"][1])

    def serialize(self, x: Any, load_dir: str | Path = ""):
        """
        Convert data from human-readable format to serialized format for a browser.
        """
        return x

    def deserialize(
        self,
        x: Any,
        save_dir: str | Path | None = None,
        root_url: str | None = None,
        hf_token: str | None = None,
    ):
        """
        Convert data from serialized format for a browser to human-readable format.
        """
        return x


class SimpleSerializable(Serializable):
    """General class that does not perform any serialization or deserialization."""

    def api_info(self) -> dict[str, bool | dict]:
        return {
            "info": serializer_types["SimpleSerializable"],
            "serialized_info": False,
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": None,
            "serialized": None,
        }


class StringSerializable(Serializable):
    """Expects a string as input/output but performs no serialization."""

    def api_info(self) -> dict[str, bool | dict]:
        return {
            "info": serializer_types["StringSerializable"],
            "serialized_info": False,
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": "Howdy!",
            "serialized": "Howdy!",
        }


class ListStringSerializable(Serializable):
    """Expects a list of strings as input/output but performs no serialization."""

    def api_info(self) -> dict[str, bool | dict]:
        return {
            "info": serializer_types["ListStringSerializable"],
            "serialized_info": False,
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": ["Howdy!", "Merhaba"],
            "serialized": ["Howdy!", "Merhaba"],
        }


class BooleanSerializable(Serializable):
    """Expects a boolean as input/output but performs no serialization."""

    def api_info(self) -> dict[str, bool | dict]:
        return {
            "info": serializer_types["BooleanSerializable"],
            "serialized_info": False,
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": True,
            "serialized": True,
        }


class NumberSerializable(Serializable):
    """Expects a number (int/float) as input/output but performs no serialization."""

    def api_info(self) -> dict[str, bool | dict]:
        return {
            "info": serializer_types["NumberSerializable"],
            "serialized_info": False,
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": 5,
            "serialized": 5,
        }


class ImgSerializable(Serializable):
    """Expects a base64 string as input/output which is serialized to a filepath."""

    def serialized_info(self):
        return {"type": "string", "description": "filepath or URL to image"}

    def api_info(self) -> dict[str, bool | dict]:
        return {"info": serializer_types["ImgSerializable"], "serialized_info": True}

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": media_data.BASE64_IMAGE,
            "serialized": "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
        }

    def serialize(
        self,
        x: str | None,
        load_dir: str | Path = "",
    ) -> str | None:
        """
        Convert from human-friendly version of a file (string filepath) to a serialized
        representation (base64).
        Parameters:
            x: String path to file to serialize
            load_dir: Path to directory containing x
        """
        if x is None or x == "":
            return None
        is_url = utils.is_valid_url(x)
        path = x if is_url else Path(load_dir) / x
        return utils.encode_url_or_file_to_base64(path)

    def deserialize(
        self,
        x: str | None,
        save_dir: str | Path | None = None,
        root_url: str | None = None,
        hf_token: str | None = None,
    ) -> str | None:
        """
        Convert from serialized representation of a file (base64) to a human-friendly
        version (string filepath). Optionally, save the file to the directory specified by save_dir
        Parameters:
            x: Base64 representation of image to deserialize into a string filepath
            save_dir: Path to directory to save the deserialized image to
            root_url: Ignored
            hf_token: Ignored
        """
        if x is None or x == "":
            return None
        file = utils.decode_base64_to_file(x, dir=save_dir)
        return file.name


class FileSerializable(Serializable):
    """Expects a dict with base64 representation of object as input/output which is serialized to a filepath."""

    def serialized_info(self):
        return self._single_file_serialized_info()

    def _single_file_api_info(self):
        return {
            "info": serializer_types["SingleFileSerializable"],
            "serialized_info": True,
        }

    def _single_file_serialized_info(self):
        return {"type": "string", "description": "filepath or URL to file"}

    def _multiple_file_serialized_info(self):
        return {
            "type": "array",
            "description": "List of filepath(s) or URL(s) to files",
            "items": {"type": "string", "description": "filepath or URL to file"},
        }

    def _multiple_file_api_info(self):
        return {
            "info": serializer_types["MultipleFileSerializable"],
            "serialized_info": True,
        }

    def api_info(self) -> dict[str, dict | bool]:
        return self._single_file_api_info()

    def example_inputs(self) -> dict[str, Any]:
        return self._single_file_example_inputs()

    def _single_file_example_inputs(self) -> dict[str, Any]:
        return {
            "raw": {"is_file": False, "data": media_data.BASE64_FILE},
            "serialized": "https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf",
        }

    def _multiple_file_example_inputs(self) -> dict[str, Any]:
        return {
            "raw": [{"is_file": False, "data": media_data.BASE64_FILE}],
            "serialized": [
                "https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf"
            ],
        }

    def _serialize_single(
        self, x: str | FileData | None, load_dir: str | Path = ""
    ) -> FileData | None:
        if x is None or isinstance(x, dict):
            return x
        if utils.is_valid_url(x):
            filename = x
            size = None
        else:
            filename = str(Path(load_dir) / x)
            size = Path(filename).stat().st_size
        return {
            "name": filename,
            "data": utils.encode_url_or_file_to_base64(filename),
            "orig_name": Path(filename).name,
            "is_file": False,
            "size": size,
        }

    def _deserialize_single(
        self,
        x: str | FileData | None,
        save_dir: str | None = None,
        root_url: str | None = None,
        hf_token: str | None = None,
    ) -> str | None:
        if x is None:
            return None
        if isinstance(x, str):
            file_name = utils.decode_base64_to_file(x, dir=save_dir).name
        elif isinstance(x, dict):
            if x.get("is_file"):
                filepath = x.get("name")
                assert filepath is not None, f"The 'name' field is missing in {x}"
                if root_url is not None:
                    file_name = utils.download_tmp_copy_of_file(
                        root_url + "file=" + filepath,
                        hf_token=hf_token,
                        dir=save_dir,
                    ).name
                else:
                    file_name = utils.create_tmp_copy_of_file(
                        filepath, dir=save_dir
                    ).name
            else:
                data = x.get("data")
                assert data is not None, f"The 'data' field is missing in {x}"
                file_name = utils.decode_base64_to_file(data, dir=save_dir).name
        else:
            raise ValueError(
                f"A FileSerializable component can only deserialize a string or a dict, not a {type(x)}: {x}"
            )
        return file_name

    def serialize(
        self,
        x: str | FileData | None | list[str | FileData | None],
        load_dir: str | Path = "",
    ) -> FileData | None | list[FileData | None]:
        """
        Convert from human-friendly version of a file (string filepath) to a
        serialized representation (base64)
        Parameters:
            x: String path to file to serialize
            load_dir: Path to directory containing x
        """
        if x is None or x == "":
            return None
        if isinstance(x, list):
            return [self._serialize_single(f, load_dir=load_dir) for f in x]
        else:
            return self._serialize_single(x, load_dir=load_dir)

    def deserialize(
        self,
        x: str | FileData | None | list[str | FileData | None],
        save_dir: Path | str | None = None,
        root_url: str | None = None,
        hf_token: str | None = None,
    ) -> str | None | list[str | None]:
        """
        Convert from serialized representation of a file (base64) to a human-friendly
        version (string filepath). Optionally, save the file to the directory specified by `save_dir`
        Parameters:
            x: Base64 representation of file to deserialize into a string filepath
            save_dir: Path to directory to save the deserialized file to
            root_url: If this component is loaded from an external Space, this is the URL of the Space.
            hf_token: If this component is loaded from an external private Space, this is the access token for the Space
        """
        if x is None:
            return None
        if isinstance(save_dir, Path):
            save_dir = str(save_dir)
        if isinstance(x, list):
            return [
                self._deserialize_single(
                    f, save_dir=save_dir, root_url=root_url, hf_token=hf_token
                )
                for f in x
            ]
        else:
            return self._deserialize_single(
                x, save_dir=save_dir, root_url=root_url, hf_token=hf_token
            )


class VideoSerializable(FileSerializable):
    def serialized_info(self):
        return {"type": "string", "description": "filepath or URL to video file"}

    def api_info(self) -> dict[str, dict | bool]:
        return {"info": serializer_types["FileSerializable"], "serialized_info": True}

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": {"is_file": False, "data": media_data.BASE64_VIDEO},
            "serialized": "https://github.com/gradio-app/gradio/raw/main/test/test_files/video_sample.mp4",
        }

    def serialize(
        self, x: str | None, load_dir: str | Path = ""
    ) -> tuple[FileData | None, None]:
        return (super().serialize(x, load_dir), None)  # type: ignore

    def deserialize(
        self,
        x: tuple[FileData | None, FileData | None] | None,
        save_dir: Path | str | None = None,
        root_url: str | None = None,
        hf_token: str | None = None,
    ) -> str | tuple[str | None, str | None] | None:
        """
        Convert from serialized representation of a file (base64) to a human-friendly
        version (string filepath). Optionally, save the file to the directory specified by `save_dir`
        """
        if isinstance(x, (tuple, list)):
            assert len(x) == 2, f"Expected tuple of length 2. Received: {x}"
            x_as_list = [x[0], x[1]]
        else:
            raise ValueError(f"Expected tuple of length 2. Received: {x}")
        deserialized_file = super().deserialize(x_as_list, save_dir, root_url, hf_token)  # type: ignore
        if isinstance(deserialized_file, list):
            return deserialized_file[0]  # ignore subtitles


class JSONSerializable(Serializable):
    def serialized_info(self):
        return {"type": "string", "description": "filepath to JSON file"}

    def api_info(self) -> dict[str, dict | bool]:
        return {"info": serializer_types["JSONSerializable"], "serialized_info": True}

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": {"a": 1, "b": 2},
            "serialized": None,
        }

    def serialize(
        self,
        x: str | None,
        load_dir: str | Path = "",
    ) -> dict | list | None:
        """
        Convert from a a human-friendly version (string path to json file) to a
        serialized representation (json string)
        Parameters:
            x: String path to json file to read to get json string
            load_dir: Path to directory containing x
        """
        if x is None or x == "":
            return None
        return utils.file_to_json(Path(load_dir) / x)

    def deserialize(
        self,
        x: str | dict | list,
        save_dir: str | Path | None = None,
        root_url: str | None = None,
        hf_token: str | None = None,
    ) -> str | None:
        """
        Convert from serialized representation (json string) to a human-friendly
        version (string path to json file).  Optionally, save the file to the directory specified by `save_dir`
        Parameters:
            x: Json string
            save_dir: Path to save the deserialized json file to
            root_url: Ignored
            hf_token: Ignored
        """
        if x is None:
            return None
        return utils.dict_or_str_to_json_file(x, dir=save_dir).name


class GallerySerializable(Serializable):
    def serialized_info(self):
        return {
            "type": "string",
            "description": "path to directory with images and a file associating images with captions called captions.json",
        }

    def api_info(self) -> dict[str, dict | bool]:
        return {
            "info": serializer_types["GallerySerializable"],
            "serialized_info": True,
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": [media_data.BASE64_IMAGE] * 2,
            "serialized": [
                "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
            ]
            * 2,
        }

    def serialize(
        self, x: str | None, load_dir: str | Path = ""
    ) -> list[list[str | None]] | None:
        if x is None or x == "":
            return None
        files = []
        captions_file = Path(x) / "captions.json"
        with captions_file.open("r") as captions_json:
            captions = json.load(captions_json)
        for file_name, caption in captions.items():
            img = FileSerializable().serialize(file_name)
            files.append([img, caption])
        return files

    def deserialize(
        self,
        x: list[list[str | None]] | None,
        save_dir: str = "",
        root_url: str | None = None,
        hf_token: str | None = None,
    ) -> None | str:
        if x is None:
            return None
        gallery_path = Path(save_dir) / str(uuid.uuid4())
        gallery_path.mkdir(exist_ok=True, parents=True)
        captions = {}
        for img_data in x:
            if isinstance(img_data, (list, tuple)):
                img_data, caption = img_data
            else:
                caption = None
            name = FileSerializable().deserialize(
                img_data, gallery_path, root_url=root_url, hf_token=hf_token
            )
            captions[name] = caption
            captions_file = gallery_path / "captions.json"
            with captions_file.open("w") as captions_json:
                json.dump(captions, captions_json)
        return os.path.abspath(gallery_path)


SERIALIZER_MAPPING = {}
for cls in Serializable.__subclasses__():
    SERIALIZER_MAPPING[cls.__name__] = cls
    for subcls in cls.__subclasses__():
        SERIALIZER_MAPPING[subcls.__name__] = subcls

SERIALIZER_MAPPING["Serializable"] = SimpleSerializable
SERIALIZER_MAPPING["File"] = FileSerializable
SERIALIZER_MAPPING["UploadButton"] = FileSerializable

COMPONENT_MAPPING: dict[str, type] = {
    "textbox": StringSerializable,
    "number": NumberSerializable,
    "slider": NumberSerializable,
    "checkbox": BooleanSerializable,
    "checkboxgroup": ListStringSerializable,
    "radio": StringSerializable,
    "dropdown": SimpleSerializable,
    "image": ImgSerializable,
    "video": FileSerializable,
    "audio": FileSerializable,
    "file": FileSerializable,
    "dataframe": JSONSerializable,
    "timeseries": JSONSerializable,
    "state": SimpleSerializable,
    "button": StringSerializable,
    "uploadbutton": FileSerializable,
    "colorpicker": StringSerializable,
    "label": JSONSerializable,
    "highlightedtext": JSONSerializable,
    "json": JSONSerializable,
    "html": StringSerializable,
    "gallery": GallerySerializable,
    "chatbot": JSONSerializable,
    "model3d": FileSerializable,
    "plot": JSONSerializable,
    "barplot": JSONSerializable,
    "lineplot": JSONSerializable,
    "scatterplot": JSONSerializable,
    "markdown": StringSerializable,
    "dataset": StringSerializable,
    "code": StringSerializable,
    "interpretation": SimpleSerializable,
    "annotatedimage": JSONSerializable,
}
