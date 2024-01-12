"""Group module of the YouFace API.
"""
from typing import List
from yk_face_api_models import Template
from yk_utils.apis import request, request_async


def create(group_id: str):
    """Create a new group with specified `group_id`.
    :param group_id:
         ID of the group to be created.
    :return:
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")

    url = f'gallery/{group_id}'
    request('POST', url)


async def create_async(group_id: str):
    """
    Create a new group with specified `group_id`.
    Performs the request asynchronously.
    :param group_id:
         ID of the group to be created.
    :return:
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")

    url = f'gallery/{group_id}'
    await request_async('POST', url)


def delete(group_id: str):
    """Delete an existing group with specified `group_id`.
    :param group_id:
         ID of the group to be deleted. `group_id` is created in `group.create`.
    :return:
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")

    url = f'gallery/{group_id}'
    request('DELETE', url)


async def delete_async(group_id: str):
    """
    Delete an existing group with specified `group_id`.
    Performs the request asynchronously.
    :param group_id:
         ID of the group to be deleted. `group_id` is created in `group.create`.
    :return:
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")

    url = f'gallery/{group_id}'
    await request_async('DELETE', url)


def list_ids(group_id: str) -> List[str]:
    """List all person ids in a specified `group_id`.
    :param group_id:
         ID of the group to be listed. `group_id` is created in `group.create`.
    :return:
        An array of person ids.
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")

    url = f'gallery/{group_id}'
    return request('GET', url)


async def list_ids_async(group_id: str) -> List[str]:
    """
    List all person ids in a specified `group_id`.
    Performs the request asynchronously.
    :param group_id:
         ID of the group to be listed. `group_id` is created in `group.create`.
    :return:
        An array of person ids.
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")

    url = f'gallery/{group_id}'
    return await request_async('GET', url)


def add_person(group_id: str, person_id: str, face_template: str):
    """Add a person to a group.
    :param group_id:
         ID of the group. `group_id` is created in `group.create`.
    :param person_id:
        Person ID.
    :param face_template:
        Biometric template to be associated with the provided `person_id` (obtained from `face.process`).
    :return:
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")
    if person_id is None:
        raise ValueError("Person ID must be specified.")

    url = f'gallery/{group_id}/{person_id}'
    template_request = Template(template=face_template).dict()
    request('POST', url, json=template_request)


async def add_person_async(group_id: str, person_id: str, face_template: str):
    """
    Add a person to a group.
    Performs the request asynchronously.
    :param group_id:
         ID of the group. `group_id` is created in `group.create`.
    :param person_id:
        Person ID.
    :param face_template:
        Biometric template to be associated with the provided `person_id` (obtained from `face.process`).
    :return:
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")
    if person_id is None:
        raise ValueError("Person ID must be specified.")

    url = f'gallery/{group_id}/{person_id}'
    template_request = Template(template=face_template).dict()
    await request_async('POST', url, json=template_request)


def get_person_template(group_id: str, person_id: str) -> str:
    """Get the biometric template of a specified `person_id` in `group_id`.
    :param group_id:
          ID of the group where the person is (used in `group.add_person`).
    :param person_id:
        Person ID. `person_id` is created in `group.add_person`.
    :return:
        The biometric template of this person.
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")
    if person_id is None:
        raise ValueError("Person ID must be specified.")

    url = f'gallery/{group_id}/{person_id}'
    json_response = request('GET', url)
    return json_response['template']


async def get_person_template_async(group_id: str, person_id: str) -> str:
    """
    Get the biometric template of a specified `person_id` in `group_id`.
    Performs the request asynchronously.
    :param group_id:
          ID of the group where the person is (used in `group.add_person`).
    :param person_id:
        Person ID. `person_id` is created in `group.add_person`.
    :return:
        The biometric template of this person.
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")
    if person_id is None:
        raise ValueError("Person ID must be specified.")

    url = f'gallery/{group_id}/{person_id}'
    json_response = await request_async('GET', url)
    return json_response['template']


def remove_person(group_id: str, person_id: str):
    """Remove a person from a group.
    :param group_id:
         ID of the group where the person is (used in `group.add_person`).
    :param person_id:
        Person ID. `person_id` is created in `group.add_person`.
    :return:
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")
    if person_id is None:
        raise ValueError("Person ID must be specified.")

    url = f'gallery/{group_id}/{person_id}'
    request('DELETE', url)


async def remove_person_async(group_id: str, person_id: str):
    """
    Remove a person from a group.
    Performs the request asynchronously.
    :param group_id:
         ID of the group where the person is (used in `group.add_person`).
    :param person_id:
        Person ID. `person_id` is created in `group.add_person`.
    :return:
    """
    if group_id is None:
        raise ValueError("Group ID must be specified.")
    if person_id is None:
        raise ValueError("Person ID must be specified.")

    url = f'gallery/{group_id}/{person_id}'
    await request_async('DELETE', url)
