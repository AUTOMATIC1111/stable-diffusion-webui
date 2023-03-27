from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from gradio import processing_utils, utils
from gradio.context import Context


class Serializable(ABC):
    @abstractmethod
    def serialize(self, x: Any, load_dir: str | Path = ""):
        """
        Convert data from human-readable format to serialized format for a browser.
        """
        pass

    @abstractmethod
    def deserialize(
        self,
        x: Any,
        save_dir: str | Path | None = None,
        root_url: str | None = None,
    ):
        """
        Convert data from serialized format for a browser to human-readable format.
        """
        pass


class SimpleSerializable(Serializable):
    def serialize(self, x: Any, load_dir: str | Path = "") -> Any:
        """
        Convert data from human-readable format to serialized format. For SimpleSerializable components, this is a no-op.
        Parameters:
            x: Input data to serialize
            load_dir: Ignored
        """
        return x

    def deserialize(
        self,
        x: Any,
        save_dir: str | Path | None = None,
        root_url: str | None = None,
    ):
        """
        Convert data from serialized format to human-readable format. For SimpleSerializable components, this is a no-op.
        Parameters:
            x: Input data to deserialize
            save_dir: Ignored
            root_url: Ignored
        """
        return x


class ImgSerializable(Serializable):
    def serialize(
        self,
        x: str | None,
        load_dir: str | Path = "",
    ) -> str | None:
        """
        Convert from human-friendly version of a file (string filepath) to a seralized
        representation (base64).
        Parameters:
            x: String path to file to serialize
            load_dir: Path to directory containing x
        """
        if x is None or x == "":
            return None
        is_url = utils.validate_url(x)
        path = x if is_url else Path(load_dir) / x
        return processing_utils.encode_url_or_file_to_base64(path)

    def deserialize(
        self,
        x: str | None,
        save_dir: str | Path | None = None,
        root_url: str | None = None,
    ) -> str | None:
        """
        Convert from serialized representation of a file (base64) to a human-friendly
        version (string filepath). Optionally, save the file to the directory specified by save_dir
        Parameters:
            x: Base64 representation of image to deserialize into a string filepath
            save_dir: Path to directory to save the deserialized image to
            root_url: Ignored
        """
        if x is None or x == "":
            return None
        file = processing_utils.decode_base64_to_file(x, dir=save_dir)
        return file.name


class FileSerializable(Serializable):
    def serialize(
        self,
        x: str | None,
        load_dir: str | Path = "",
    ) -> Dict | None:
        """
        Convert from human-friendly version of a file (string filepath) to a
        seralized representation (base64)
        Parameters:
            x: String path to file to serialize
            load_dir: Path to directory containing x
        """
        if x is None or x == "":
            return None
        filename = str(Path(load_dir) / x)
        return {
            "name": filename,
            "data": processing_utils.encode_url_or_file_to_base64(filename),
            "orig_name": Path(filename).name,
            "is_file": False,
        }

    def deserialize(
        self,
        x: str | Dict | None,
        save_dir: Path | str | None = None,
        root_url: str | None = None,
    ) -> str | None:
        """
        Convert from serialized representation of a file (base64) to a human-friendly
        version (string filepath). Optionally, save the file to the directory specified by `save_dir`
        Parameters:
            x: Base64 representation of file to deserialize into a string filepath
            save_dir: Path to directory to save the deserialized file to
            root_url: If this component is loaded from an external Space, this is the URL of the Space
        """
        if x is None:
            return None
        if isinstance(save_dir, Path):
            save_dir = str(save_dir)
        if isinstance(x, str):
            file_name = processing_utils.decode_base64_to_file(x, dir=save_dir).name
        elif isinstance(x, dict):
            if x.get("is_file", False):
                if root_url is not None:
                    file_name = processing_utils.download_tmp_copy_of_file(
                        root_url + "file=" + x["name"],
                        access_token=Context.access_token,
                        dir=save_dir,
                    ).name
                else:
                    file_name = processing_utils.create_tmp_copy_of_file(
                        x["name"], dir=save_dir
                    ).name
            else:
                file_name = processing_utils.decode_base64_to_file(
                    x["data"], dir=save_dir
                ).name
        else:
            raise ValueError(
                f"A FileSerializable component cannot only deserialize a string or a dict, not a: {type(x)}"
            )
        return file_name


class JSONSerializable(Serializable):
    def serialize(
        self,
        x: str | None,
        load_dir: str | Path = "",
    ) -> Dict | None:
        """
        Convert from a a human-friendly version (string path to json file) to a
        serialized representation (json string)
        Parameters:
            x: String path to json file to read to get json string
            load_dir: Path to directory containing x
        """
        if x is None or x == "":
            return None
        return processing_utils.file_to_json(Path(load_dir) / x)

    def deserialize(
        self,
        x: str | Dict,
        save_dir: str | Path | None = None,
        root_url: str | None = None,
    ) -> str | None:
        """
        Convert from serialized representation (json string) to a human-friendly
        version (string path to json file).  Optionally, save the file to the directory specified by `save_dir`
        Parameters:
            x: Json string
            save_dir: Path to save the deserialized json file to
            root_url: Ignored
        """
        if x is None:
            return None
        return processing_utils.dict_or_str_to_json_file(x, dir=save_dir).name
