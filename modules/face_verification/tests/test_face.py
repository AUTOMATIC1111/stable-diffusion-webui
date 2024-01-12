""" Face Integration Tests """
import os
import string
import random
import asyncio
import pytest
import yk_face as YKF
from yk_utils.apis import YoonikApiException

from yk_face_api_models import ProcessRequestConfig

BASE_URL = os.getenv('YK_FACE_BASE_URL')
YKF.BaseUrl.set(BASE_URL)

KEY = os.getenv('YK_FACE_X_API_KEY')
YKF.Key.set(KEY)


__image_file = './sample/detection1.jpg'
__person_id = "person1"
__template: str

__group_id_async = "test_face_sdk_py_group_async"
__group_id = "test_face_sdk_py_group"


def random_str(length: int = 190) -> str:
    """
    Generate random string
    :param length: length of the string to create
    :return:
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.parametrize('use_async', [(True,), (False,)])
@pytest.mark.parametrize('configurations', [
    [
        ProcessRequestConfig(name="config1", value="12490812.523"),
        ProcessRequestConfig(name="config1.1", value="stringvalue"),
        ProcessRequestConfig(name="config2", bvalue=True),
        ProcessRequestConfig(name="config3", bvalue=False),
     ]
])
def test_face_process_with_valid_image(
        use_async: bool,
        configurations,
        loop: asyncio.AbstractEventLoop
):
    """
    Test sync and async valid process request.
    :param use_async: flag to use the async function
    :param loop: event loop for the current test
    :return:
    """
    if use_async:
        response = loop.run_until_complete(
            YKF.face.process_async(
                __image_file,
                configurations=configurations
            )
        )
    else:
        response = YKF.face.process(__image_file)

    assert type(response) == list
    detected_face: dict = response[0]
    assert type(detected_face) == dict
    assert detected_face.get("biometric_type")
    assert detected_face.get("x")
    assert detected_face.get("y")
    assert detected_face.get("quality_metrics")
    assert detected_face.get("biometric_points")
    assert detected_face.get("template")
    global __template
    __template = detected_face.get("template")


@pytest.mark.parametrize('use_async', [(True,), (False,)])
@pytest.mark.parametrize('configurations', [
    [
        ProcessRequestConfig(name="config1", value="12490812.523"),
        ProcessRequestConfig(name="config1.1", value="stringvalue"),
        ProcessRequestConfig(name="config2", bvalue=True),
        ProcessRequestConfig(name="config3", bvalue=False),
     ]
])
def test_face_process_with_invalid_image(
        use_async: bool,
        configurations,
        loop: asyncio.AbstractEventLoop
):
    """
    Test sync and async invalid process request.
    :param use_async: flag to use the async function
    :param loop: event loop for the current test
    :return:
    """
    with pytest.raises(YoonikApiException) as exception:
        if use_async:
            loop.run_until_complete(YKF.face.process_async(random_str(),
                                                           configurations=configurations))
        else:
            YKF.face.process(random_str())
    assert exception.value.status_code == 409


@pytest.mark.parametrize('use_async', [(True,), (False,)])
def test_face_verify_with_valid_templates(use_async: bool, loop: asyncio.AbstractEventLoop):
    """
    Test sync and async valid verify request.
    :param use_async: flag to use the async function
    :param loop: event loop for the current test
    :return:
    """
    if use_async:
        response = loop.run_until_complete(
            YKF.face.verify_async(
                __template,
                __template
            )
        )
    else:
        response = YKF.face.verify(__template, __template)

    assert type(response) == float
    assert response > 0


@pytest.mark.parametrize('use_async', [(True,), (False,)])
def test_face_verify_with_invalid_templates(use_async: bool, loop: asyncio.AbstractEventLoop):
    """
    Test sync and async invalid verify request.
    :param use_async: flag to use the async function
    :param loop: event loop for the current test
    :return:
    """
    with pytest.raises(YoonikApiException) as exception:
        if use_async:
            loop.run_until_complete(
                YKF.face.verify_async(
                    random_str(),
                    random_str()
                )
            )
        else:
            YKF.face.verify(random_str(), random_str())

    assert exception.value.status_code == 409


@pytest.mark.parametrize('use_async, group_id', [
    (True, __group_id_async),
    (False, __group_id)
])
def test_group_create(use_async: bool, group_id: str, loop: asyncio.AbstractEventLoop):
    """
    Test sync and async valid group create request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
        Passes if no exception is raised.
    """
    try:
        if use_async:
            loop.run_until_complete(YKF.group.create_async(group_id))
        else:
            YKF.group.create(group_id)
    except YoonikApiException:
        assert False


@pytest.mark.parametrize('use_async, group_id', [
    (True, __group_id_async),
    (False, __group_id)
])
def test_group_add_person(use_async: bool, group_id: str, loop: asyncio.AbstractEventLoop):
    """
    Test sync and async valid group add_person request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
        Passes if no exception is raised.
    """
    try:
        if use_async:
            loop.run_until_complete(
                YKF.group.add_person_async(
                    group_id=group_id,
                    person_id=__person_id,
                    face_template=__template
                )
            )
        else:
            YKF.group.add_person(
                group_id=group_id,
                person_id=__person_id,
                face_template=__template
            )
    except YoonikApiException:
        assert False


@pytest.mark.parametrize('use_async, group_id', [
    (True, 'invalid_group1'),
    (False, 'invalid_group3')
])
def test_group_add_person_to_invalid_group(
        use_async: bool,
        group_id: str,
        loop: asyncio.AbstractEventLoop
):
    """
    Test sync and async invalid group add_person request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
        Passes if YoonikApiException is raised
    """
    with pytest.raises(YoonikApiException) as exception:
        if use_async:
            loop.run_until_complete(
                YKF.group.add_person_async(
                    group_id,
                    person_id=__person_id,
                    face_template=__template
                )
            )
        else:
            YKF.group.add_person(
                group_id,
                person_id=__person_id,
                face_template=__template
            )

    assert exception.value.status_code == 404


@pytest.mark.parametrize('use_async, group_id', [
    (True, __group_id_async),
    (False, __group_id)
])
def test_group_list_ids(use_async: bool, group_id: str, loop: asyncio.AbstractEventLoop):
    """
    Test sync and async valid group list_ids request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
    """
    if use_async:
        list_ids = loop.run_until_complete(
            YKF.group.list_ids_async(group_id)
        )
    else:
        list_ids = YKF.group.list_ids(group_id)

    assert type(list_ids) is list
    assert len(list_ids) == 1
    assert __person_id in list_ids


@pytest.mark.parametrize('use_async, group_id', [
    (True, 'invalidgroup1'),
    (True, 'notvalidgroup2'),
    (False, 'notvalidgroup5'),
    (False, 'invalidgroup11')
])
def test_group_list_ids_with_invalid_group(
        use_async: bool,
        group_id: str,
        loop: asyncio.AbstractEventLoop):
    """
    Test sync and async invalid group list_ids request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
    """
    with pytest.raises(YoonikApiException) as exception:
        if use_async:
            loop.run_until_complete(YKF.group.list_ids_async(group_id))
        else:
            YKF.group.list_ids(group_id)
    assert exception.value.status_code == 404


@pytest.mark.parametrize('use_async, group_id', [
    (True, __group_id_async),
    (False, __group_id)
])
def test_face_verify_id(
        use_async: bool,
        group_id: str,
        loop: asyncio.AbstractEventLoop):
    """
    Test sync and async face verify_id request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
    """
    if use_async:
        verify_score = loop.run_until_complete(
            YKF.face.verify_id_async(
                face_template=__template,
                person_id=__person_id,
                group_id=group_id
            )
        )
    else:
        verify_score = YKF.face.verify_id(
            face_template=__template,
            person_id=__person_id,
            group_id=group_id
        )
    assert type(verify_score) == float
    assert verify_score > 0


@pytest.mark.parametrize('use_async, group_id', [
    (True, __group_id_async),
    (False, __group_id)
])
def test_face_identify(
        use_async: bool,
        group_id: str,
        loop: asyncio.AbstractEventLoop):
    """
    Test sync and async face identify request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
    """
    if use_async:
        identified = loop.run_until_complete(
            YKF.face.identify_async(
                face_template=__template,
                group_id=group_id,
                candidate_list_length=3
            )
        )
    else:
        identified = YKF.face.identify(
                face_template=__template,
                group_id=group_id,
                candidate_list_length=3
            )

    assert type(identified) == list
    assert len(identified) == 1

    assert type(identified[0]["template_id"]) == str
    assert identified[0]["template_id"] == __person_id

    assert type(identified[0]["score"]) == float
    assert identified[0]["score"] > 0


@pytest.mark.parametrize('use_async, group_id', [
    (True, __group_id_async),
    (False, __group_id)
])
def test_group_remove_person(
        use_async: bool,
        group_id: str,
        loop: asyncio.AbstractEventLoop):
    """
    Test sync and async group remove person request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
        It passes if no YoonikApiException is raised
    """
    try:
        if use_async:
            loop.run_until_complete(
                YKF.group.remove_person_async(
                    person_id=__person_id,
                    group_id=group_id,
                )
            )
        else:
            YKF.group.remove_person(
                    person_id=__person_id,
                    group_id=group_id,
                )
    except YoonikApiException:
        assert False


@pytest.mark.parametrize('use_async, group_id', [
    (True, __group_id_async),
    (False, __group_id)
])
def test_group_delete(
        use_async: bool,
        group_id: str,
        loop: asyncio.AbstractEventLoop):
    """
    Test sync and async group delete request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
        It passes if no YoonikApiException is raised
    """
    try:
        if use_async:
            loop.run_until_complete(
                YKF.group.delete_async(
                    group_id=group_id,
                )
            )
        else:
            YKF.group.delete(group_id=group_id)
    except YoonikApiException:
        assert False


@pytest.mark.parametrize('use_async, group_id', [
    (True, "invalidGroupName2"),
    (False, "notValidGroupName")
])
def test_group_delete_invalid_group_id(
        use_async: bool,
        group_id: str,
        loop: asyncio.AbstractEventLoop):
    """
    Test sync and async invalid group delete request.
    :param use_async: flag to use the async function
    :param group_id: group identifier
    :param loop: event loop for the current test
    :return:
        It passes if YoonikApiException is raised with a 404 status code
    """
    with pytest.raises(YoonikApiException) as exception:
        if use_async:
            loop.run_until_complete(
                YKF.group.delete_async(
                    group_id=group_id,
                )
            )
        else:
            YKF.group.delete(group_id=group_id)

    assert exception.value.status_code == 404


@pytest.mark.parametrize('use_async', [(True,), (False,)])
def test_verify_images(
        use_async: bool,
        loop: asyncio.AbstractEventLoop):
    """
    Test sync and async valid verify_images request.
    :param use_async: flag to use the async function
    :param loop: event loop for the current test
    :return:
    """
    if use_async:
        verify_score = loop.run_until_complete(
            YKF.face.verify_images_async(
                __image_file,
                __image_file
            )
        )
    else:
        verify_score = YKF.face.verify_images(__image_file, __image_file)

    assert type(verify_score) == float
    assert verify_score > 0


@pytest.mark.parametrize('use_async', [(True,), (False,)])
def test_verify_images_with_invalid_images(
        use_async: bool,
        loop: asyncio.AbstractEventLoop):
    """
    Test sync and async invalid verify_images request.
    :param use_async: flag to use the async function
    :param loop: event loop for the current test
    :return:
        It passes if FaceApiException is raised
    """
    with pytest.raises(YoonikApiException) as exception:
        if use_async:
            loop.run_until_complete(
                YKF.face.verify_images_async(
                    random_str(),
                    random_str()
                )
            )
        else:
            YKF.face.verify_images(random_str(), random_str())

    assert exception.value.status_code == 409
