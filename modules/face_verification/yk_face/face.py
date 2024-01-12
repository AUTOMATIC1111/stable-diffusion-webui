"""Face module of the YouFace API.
"""
import asyncio
from typing import List, Dict
from yk_utils.images import parse_image
from yk_utils.apis import request, request_async
from yk_face_api_models import ProcessRequest, VerifyRequest, VerifyIdRequest, IdentifyRequest, \
    ProcessRequestConfig
from yk_face.util import FaceException, face_process_validation


class FaceRouterEndpoints:
    process = "face/process"
    verify = "face/verify"
    verify_id = "face/verify_id"
    identify = "face/identify"


def __process_request_validation(
        image,
        processings: List[str] = None,
        configurations: List[ProcessRequestConfig] = None
) -> Dict:
    """ Validates the process endpoint request.
        :param configurations:
            A list of ProcessRequestConfig, for dynamic configurations.
        :param image:
            A base64 string or a file path or a file-like object representing an image.
        :param processings:
            List of desired processings (if None, it will perform all processings):
                'detect'   - Perform face and landmarks detection.
                'analyze'  - Perform quality analysis (brightness, contrast, sharpness, etc).
                'templify' - Perform template extraction.
        :return:
            dictionary of the payload to be sent in HTTP Request
        :raises:
            ValueError if image is not provided.
    """
    if image is None:
        raise ValueError("image must be provided")

    image_b64 = parse_image(image)
    if processings is None:
        processings = ['detect', 'analyze', 'templify']
    process_request = ProcessRequest(
        image=image_b64,
        processings=processings,
        configuration=configurations
    ).dict()
    return process_request


def process(
        image,
        processings: List[str] = None,
        configurations: List[ProcessRequestConfig] = None
) -> List[Dict]:
    """Process human faces in an image.
    :param configurations:
        A list of ProcessRequestConfig, for dynamic configurations.
    :param image:
        A base64 string or a file path or a file-like object representing an image.
    :param processings:
        List of desired processings (if None, it will perform all processings):
            'detect'   - Perform face and landmarks detection.
            'analyze'  - Perform quality analysis (brightness, contrast, sharpness, etc).
            'templify' - Perform template extraction.
    :return:
        List of face entries in json format.
    :raises:
        ValueError if image is not provided.
    """
    process_request = __process_request_validation(image, processings, configurations)
    return request('POST', FaceRouterEndpoints.process, json=process_request)


async def process_async(
        image,
        processings: List[str] = None,
        configurations: List[ProcessRequestConfig] = None,
) -> List[Dict]:
    """
    Process human faces in an image.
    Performs the request asynchronously.
    :param configurations:
        A list of ProcessRequestConfig, for dynamic configurations.
    :param image:
        A base64 string or a file path or a file-like object representing an image.
    :param processings:
        List of desired processings (if None, it will perform all processings):
            'detect'   - Perform face and landmarks detection.
            'analyze'  - Perform quality analysis (brightness, contrast, sharpness, etc).
            'templify' - Perform template extraction.
    :return:
        List of face entries in json format.
    :raises:
        ValueError if image is not provided.
    """
    process_request = __process_request_validation(image, processings, configurations)
    return await request_async('POST', FaceRouterEndpoints.process, json=process_request)


def verify(face_template: str, another_face_template: str) -> float:
    """Verify whether two faces belong to the same person.
    :param face_template:
        Biometric template of one face (obtained from `face.process`).
    :param another_face_template:
        Biometric template of another face (obtained from `face.process`).
    :return:
        The matching score.
    """
    verify_request = VerifyRequest(
        first_template=face_template,
        second_template=another_face_template
    ).dict()
    json_response = request('POST', FaceRouterEndpoints.verify, json=verify_request)
    return float(json_response['score'])


async def verify_async(face_template: str, another_face_template: str) -> float:
    """
    Verify whether two faces belong to the same person.
    Performs the request asynchronously.
    :param face_template:
        Biometric template of one face (obtained from `face.process`).
    :param another_face_template:
        Biometric template of another face (obtained from `face.process`).
    :return:
        The matching score.
    """
    verify_request = VerifyRequest(
        first_template=face_template,
        second_template=another_face_template
    ).dict()
    json_response = await request_async('POST', FaceRouterEndpoints.verify, json=verify_request)
    return float(json_response['score'])


def verify_id(face_template: str, person_id: str, group_id: str) -> float:
    """Verify whether one face belongs to a person.
    :param face_template:
        Biometric template of one face (obtained from `face.process`).
    :param person_id:
        Specify a certain person in a group. `person_id` is created in `group.add_person`.
    :param group_id:
        Specify a certain group where the person is. `group_id` is created in `group.create`.
    :return:
        The matching score.
    """
    verify_id_request = VerifyIdRequest(
        template=face_template,
        template_id=person_id,
        gallery_id=group_id
    ).dict()
    json_response = request('POST', FaceRouterEndpoints.verify_id, json=verify_id_request)
    return float(json_response['score'])


async def verify_id_async(face_template: str, person_id: str, group_id: str) -> float:
    """
    Verify whether one face belongs to a person.
    Performs the request asynchronously.
    :param face_template:
        Biometric template of one face (obtained from `face.process`).
    :param person_id:
        Specify a certain person in a group. `person_id` is created in `group.add_person`.
    :param group_id:
        Specify a certain group where the person is. `group_id` is created in `group.create`.
    :return:
        The matching score.
    """
    verify_id_request = VerifyIdRequest(
        template=face_template,
        template_id=person_id,
        gallery_id=group_id
    ).dict()

    json_response = await request_async(
        'POST',
        FaceRouterEndpoints.verify_id,
        json=verify_id_request
    )
    return float(json_response['score'])


def identify(
        face_template: str,
        group_id: str,
        minimum_score: float = -1.0,
        candidate_list_length: int = 1) -> List[Dict]:
    """Identify an unknown face in a group.
    :param face_template:
        Biometric template of the face to be identified (obtained from `face.process`).
    :param group_id:
        Specify a certain group to perform the identification. `group_id` is created in `group.create`.
    :param minimum_score:
        Minimum matching score for candidates.
    :param candidate_list_length:
        Maximum length of the list of resulting candidates.
    :return:
        The identified candidates for the provided face template.
    """
    identify_request = IdentifyRequest(
        template=face_template,
        candidate_list_length=candidate_list_length,
        minimum_score=minimum_score,
        gallery_id=group_id
    ).dict()
    return request('POST', FaceRouterEndpoints.identify, json=identify_request)


async def identify_async(
        face_template: str,
        group_id: str,
        minimum_score: float = -1.0,
        candidate_list_length: int = 1) -> List[Dict]:
    """
    Identify an unknown face in a group.
    Performs the request asynchronously.
    :param face_template:
        Biometric template of the face to be identified (obtained from `face.process`).
    :param group_id:
        Specify a certain group to perform the identification. `group_id` is created in `group.create`.
    :param minimum_score:
        Minimum matching score for candidates.
    :param candidate_list_length:
        Maximum length of the list of resulting candidates.
    :return:
        The identified candidates for the provided face template.
    """
    identify_request = IdentifyRequest(
        template=face_template,
        candidate_list_length=candidate_list_length,
        minimum_score=minimum_score,
        gallery_id=group_id
    ).dict()
    return await request_async('POST', FaceRouterEndpoints.identify, json=identify_request)


def verify_images(first_image, second_image) -> float:
    """
        Verifies if the face detected on the first image matches to the
        face detected on the second image.
    :param first_image:
        A base64 string or a file path or a file-like object representing an image.
    :param second_image:
        A base64 string or a file path or a file-like object representing an image.
    :return:
        Matching Score.
    """
    first_face = process(first_image)
    second_face = process(second_image)

    error_message = face_process_validation(first_face)
    if error_message:
        raise FaceException(f"First image: {error_message}")

    error_message = face_process_validation(second_face)
    if error_message:
        raise FaceException(f"Second image: {error_message}")

    return verify(first_face[0]["template"], second_face[0]["template"])


async def verify_images_async(first_image, second_image) -> float:
    """
        Verifies if the face detected on the first image matches to the
        face detected on the second image.
        Performs the requests asynchronously.
    :param first_image:
        A base64 string or a file path or a file-like object representing an image.
    :param second_image:
        A base64 string or a file path or a file-like object representing an image.
    :return:
        Matching Score.
    """
    responses = await asyncio.gather(
        process_async(first_image),
        process_async(second_image)
    )
    first_face, second_face = responses[0], responses[1]

    error_message = face_process_validation(first_face)
    if error_message:
        raise FaceException(f"First image: {error_message}")

    error_message = face_process_validation(second_face)
    if error_message:
        raise FaceException(f"Second image: {error_message}")

    return await verify_async(first_face[0]["template"], second_face[0]["template"])
