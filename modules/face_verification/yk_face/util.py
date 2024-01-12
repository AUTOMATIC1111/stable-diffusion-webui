""" Utilities for the Python SDK of the YouFace API.
"""
import yk_utils.apis


class Key:
    """Manage Subscription Key."""
    @classmethod
    def set(cls, key: str):
        """Set the Subscription Key.
        :param key:
        :return:
        """
        yk_utils.apis.Key.set(key)


class BaseUrl:
    """Manage YouFace API base URL."""
    @classmethod
    def set(cls, base_url: str):
        yk_utils.apis.BaseUrl.set(base_url)


class FaceException(Exception):
    """ YouFace Exception """
    pass


def face_process_validation(face_process: list) -> str:
    """
        Checks if the face process returned object has indeed an object (i.e it has a detected face)
        and if it only has one object (i.e only one face was detected)
    :param face_process:
        face.process return object
    :return:
        If an error is detected the corresponding message is returned.
        If no errors are detected an empty string is returned.
    """
    if not face_process:
        return "face not detected"
    if len(face_process) > 1:
        return "multiple faces detected"
    return ""
