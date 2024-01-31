# Copyright (c) OpenMMLab. All rights reserved.

from io import StringIO

from .file_client import FileClient


def list_from_file(filename,
                   prefix='',
                   offset=0,
                   max_num=0,
                   encoding='utf-8',
                   file_client_args=None):
    """Load a text file and parse the content as a list of strings.

    Note:
        In v1.3.16 and later, ``list_from_file`` supports loading a text file
        which can be storaged in different backends and parsing the content as
        a list for strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the beginning of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.
        encoding (str): Encoding used to open the file. Default utf-8.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> list_from_file('/path/of/your/file')  # disk
        ['hello', 'world']
        >>> list_from_file('s3://path/of/your/file')  # ceph or petrel
        ['hello', 'world']

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    file_client = FileClient.infer_client(file_client_args, filename)
    with StringIO(file_client.get_text(filename, encoding)) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list


def dict_from_file(filename,
                   key_type=str,
                   encoding='utf-8',
                   file_client_args=None):
    """Load a text file and parse the content as a dict.

    Each line of the text file will be two or more columns split by
    whitespaces or tabs. The first column will be parsed as dict keys, and
    the following columns will be parsed as dict values.

    Note:
        In v1.3.16 and later, ``dict_from_file`` supports loading a text file
        which can be storaged in different backends and parsing the content as
        a dict.

    Args:
        filename(str): Filename.
        key_type(type): Type of the dict keys. str is user by default and
            type conversion will be performed if specified.
        encoding (str): Encoding used to open the file. Default utf-8.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> dict_from_file('/path/of/your/file')  # disk
        {'key1': 'value1', 'key2': 'value2'}
        >>> dict_from_file('s3://path/of/your/file')  # ceph or petrel
        {'key1': 'value1', 'key2': 'value2'}

    Returns:
        dict: The parsed contents.
    """
    mapping = {}
    file_client = FileClient.infer_client(file_client_args, filename)
    with StringIO(file_client.get_text(filename, encoding)) as f:
        for line in f:
            items = line.rstrip('\n').split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
