# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import os
import os.path as osp
import re
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union
from urllib.request import urlopen

import annotator.mmpkg.mmcv as mmcv
from annotator.mmpkg.mmcv.utils.misc import has_method
from annotator.mmpkg.mmcv.utils.path import is_filepath


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    _allow_symlink = False

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def allow_symlink(self):
        return self._allow_symlink

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class CephBackend(BaseStorageBackend):
    """Ceph storage backend (for internal use).

    Args:
        path_mapping (dict|None): path mapping dict from local path to Petrel
            path. When ``path_mapping={'src': 'dst'}``, ``src`` in ``filepath``
            will be replaced by ``dst``. Default: None.

    .. warning::
        :class:`mmcv.fileio.file_client.CephBackend` will be deprecated,
        please use :class:`mmcv.fileio.file_client.PetrelBackend` instead.
    """

    def __init__(self, path_mapping=None):
        try:
            import ceph
        except ImportError:
            raise ImportError('Please install ceph to enable CephBackend.')

        warnings.warn(
            'CephBackend will be deprecated, please use PetrelBackend instead')
        self._client = ceph.S3Client()
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def get(self, filepath):
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError


class PetrelBackend(BaseStorageBackend):
    """Petrel storage backend (for internal use).

    PetrelBackend supports reading and writing data to multiple clusters.
    If the file path contains the cluster name, PetrelBackend will read data
    from specified cluster or write data to it. Otherwise, PetrelBackend will
    access the default cluster.

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Petrel path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Default: None.
        enable_mc (bool, optional): Whether to enable memcached support.
            Default: True.

    Examples:
        >>> filepath1 = 's3://path/of/file'
        >>> filepath2 = 'cluster-name:s3://path/of/file'
        >>> client = PetrelBackend()
        >>> client.get(filepath1)  # get data from default cluster
        >>> client.get(filepath2)  # get data from 'cluster-name' cluster
    """

    def __init__(self,
                 path_mapping: Optional[dict] = None,
                 enable_mc: bool = True):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')

        self._client = client.Client(enable_mc=enable_mc)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of petrel oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def get(self, filepath: Union[str, Path]) -> memoryview:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            memoryview: A memory view of expected bytes object to avoid
                copying. The memoryview object can be converted to bytes by
                ``value_buf.tobytes()``.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Save data to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        self._client.put(filepath, obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        """Save data to a given ``filepath``.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to encode the ``obj``.
                Default: 'utf-8'.
        """
        self.put(bytes(obj, encoding=encoding), filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.
        """
        if not has_method(self._client, 'delete'):
            raise NotImplementedError(
                ('Current version of Petrel Python SDK has not supported '
                 'the `delete` method, please use a higher version or dev'
                 ' branch instead.'))

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        self._client.delete(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        if not (has_method(self._client, 'contains')
                and has_method(self._client, 'isdir')):
            raise NotImplementedError(
                ('Current version of Petrel Python SDK has not supported '
                 'the `contains` and `isdir` methods, please use a higher'
                 'version or dev branch instead.'))

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        return self._client.contains(filepath) or self._client.isdir(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
                ``False`` otherwise.
        """
        if not has_method(self._client, 'isdir'):
            raise NotImplementedError(
                ('Current version of Petrel Python SDK has not supported '
                 'the `isdir` method, please use a higher version or dev'
                 ' branch instead.'))

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
                otherwise.
        """
        if not has_method(self._client, 'contains'):
            raise NotImplementedError(
                ('Current version of Petrel Python SDK has not supported '
                 'the `contains` method, please use a higher version or '
                 'dev branch instead.'))

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        return self._client.contains(filepath)

    def join_path(self, filepath: Union[str, Path],
                  *filepaths: Union[str, Path]) -> str:
        """Concatenate all file paths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result after concatenation.
        """
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith('/'):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_paths.append(self._format_path(self._map_path(path)))
        return '/'.join(formatted_paths)

    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]) -> Iterable[str]:
        """Download a file from ``filepath`` and return a temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str | Path): Download a file from ``filepath``.

        Examples:
            >>> client = PetrelBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with client.get_local_path('s3://path/of/your/file') as path:
            ...     # do something here

        Yields:
            Iterable[str]: Only yield one temporary path.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        if not has_method(self._client, 'list'):
            raise NotImplementedError(
                ('Current version of Petrel Python SDK has not supported '
                 'the `list` method, please use a higher version or dev'
                 ' branch instead.'))

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith('/'):
            dir_path += '/'

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith('/'):  # a directory path
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root):-1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(next_dir_path, list_dir,
                                                     list_file, suffix,
                                                     recursive)
                else:  # a file path
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root):]
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)


class MemcachedBackend(BaseStorageBackend):
    """Memcached storage backend.

    Attributes:
        server_list_cfg (str): Config file for memcached server list.
        client_cfg (str): Config file for memcached client.
        sys_path (str | None): Additional path to be appended to `sys.path`.
            Default: None.
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError(
                'Please install memcached to enable MemcachedBackend.')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(self.server_list_cfg,
                                                      self.client_cfg)
        # mc.pyvector servers as a point which points to a memory cache
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError


class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
        db_path (str): Lmdb database path.
        readonly (bool, optional): Lmdb environment parameter. If True,
            disallow any write operations. Default: True.
        lock (bool, optional): Lmdb environment parameter. If False, when
            concurrent access occurs, do not lock the database. Default: False.
        readahead (bool, optional): Lmdb environment parameter. If False,
            disable the OS filesystem readahead mechanism, which may improve
            random read performance when a database is larger than RAM.
            Default: False.

    Attributes:
        db_path (str): Lmdb database path.
    """

    def __init__(self,
                 db_path,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')

        self.db_path = str(db_path)
        self._client = lmdb.open(
            self.db_path,
            readonly=readonly,
            lock=lock,
            readahead=readahead,
            **kwargs)

    def get(self, filepath):
        """Get values according to the filepath.

        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
        """
        filepath = str(filepath)
        with self._client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    _allow_symlink = True

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        mmcv.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, 'wb') as f:
            f.write(obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        mmcv.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.
        """
        os.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return osp.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
                ``False`` otherwise.
        """
        return osp.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
                otherwise.
        """
        return osp.isfile(filepath)

    def join_path(self, filepath: Union[str, Path],
                  *filepaths: Union[str, Path]) -> str:
        """Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        return osp.join(filepath, *filepaths)

    @contextmanager
    def get_local_path(
            self, filepath: Union[str, Path]) -> Iterable[Union[str, Path]]:
        """Only for unified API and do nothing."""
        yield filepath

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        if list_dir and suffix is not None:
            raise TypeError('`suffix` should be None when `list_dir` is True')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith('.') and entry.is_file():
                    rel_path = osp.relpath(entry.path, root)
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif osp.isdir(entry.path):
                    if list_dir:
                        rel_dir = osp.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(entry.path, list_dir,
                                                     list_file, suffix,
                                                     recursive)

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)


class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def get(self, filepath):
        value_buf = urlopen(filepath).read()
        return value_buf

    def get_text(self, filepath, encoding='utf-8'):
        value_buf = urlopen(filepath).read()
        return value_buf.decode(encoding)

    @contextmanager
    def get_local_path(self, filepath: str) -> Iterable[str]:
        """Download a file from ``filepath``.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> client = HTTPBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with client.get_local_path('http://path/of/your/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)


class FileClient:
    """A general file client to access files in different backends.

    The client loads a file or text in a specified backend from its path
    and returns it as a binary or text file. There are two ways to choose a
    backend, the name of backend and the prefix of path. Although both of them
    can be used to choose a storage backend, ``backend`` has a higher priority
    that is if they are all set, the storage backend will be chosen by the
    backend argument. If they are all `None`, the disk backend will be chosen.
    Note that It can also register other backend accessor with a given name,
    prefixes, and backend class. In addition, We use the singleton pattern to
    avoid repeated object creation. If the arguments are the same, the same
    object will be returned.

    Args:
        backend (str, optional): The storage backend type. Options are "disk",
            "ceph", "memcached", "lmdb", "http" and "petrel". Default: None.
        prefix (str, optional): The prefix of the registered storage backend.
            Options are "s3", "http", "https". Default: None.

    Examples:
        >>> # only set backend
        >>> file_client = FileClient(backend='petrel')
        >>> # only set prefix
        >>> file_client = FileClient(prefix='s3')
        >>> # set both backend and prefix but use backend to choose client
        >>> file_client = FileClient(backend='petrel', prefix='s3')
        >>> # if the arguments are the same, the same object is returned
        >>> file_client1 = FileClient(backend='petrel')
        >>> file_client1 is file_client
        True

    Attributes:
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'disk': HardDiskBackend,
        'ceph': CephBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
        'petrel': PetrelBackend,
        'http': HTTPBackend,
    }
    # This collection is used to record the overridden backends, and when a
    # backend appears in the collection, the singleton pattern is disabled for
    # that backend, because if the singleton pattern is used, then the object
    # returned will be the backend before overwriting
    _overridden_backends = set()
    _prefix_to_backends = {
        's3': PetrelBackend,
        'http': HTTPBackend,
        'https': HTTPBackend,
    }
    _overridden_prefixes = set()

    _instances = {}

    def __new__(cls, backend=None, prefix=None, **kwargs):
        if backend is None and prefix is None:
            backend = 'disk'
        if backend is not None and backend not in cls._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(cls._backends.keys())}')
        if prefix is not None and prefix not in cls._prefix_to_backends:
            raise ValueError(
                f'prefix {prefix} is not supported. Currently supported ones '
                f'are {list(cls._prefix_to_backends.keys())}')

        # concatenate the arguments to a unique key for determining whether
        # objects with the same arguments were created
        arg_key = f'{backend}:{prefix}'
        for key, value in kwargs.items():
            arg_key += f':{key}:{value}'

        # if a backend was overridden, it will create a new object
        if (arg_key in cls._instances
                and backend not in cls._overridden_backends
                and prefix not in cls._overridden_prefixes):
            _instance = cls._instances[arg_key]
        else:
            # create a new object and put it to _instance
            _instance = super().__new__(cls)
            if backend is not None:
                _instance.client = cls._backends[backend](**kwargs)
            else:
                _instance.client = cls._prefix_to_backends[prefix](**kwargs)

            cls._instances[arg_key] = _instance

        return _instance

    @property
    def name(self):
        return self.client.name

    @property
    def allow_symlink(self):
        return self.client.allow_symlink

    @staticmethod
    def parse_uri_prefix(uri: Union[str, Path]) -> Optional[str]:
        """Parse the prefix of a uri.

        Args:
            uri (str | Path): Uri to be parsed that contains the file prefix.

        Examples:
            >>> FileClient.parse_uri_prefix('s3://path/of/your/file')
            's3'

        Returns:
            str | None: Return the prefix of uri if the uri contains '://'
                else ``None``.
        """
        assert is_filepath(uri)
        uri = str(uri)
        if '://' not in uri:
            return None
        else:
            prefix, _ = uri.split('://')
            # In the case of PetrelBackend, the prefix may contains the cluster
            # name like clusterName:s3
            if ':' in prefix:
                _, prefix = prefix.split(':')
            return prefix

    @classmethod
    def infer_client(cls,
                     file_client_args: Optional[dict] = None,
                     uri: Optional[Union[str, Path]] = None) -> 'FileClient':
        """Infer a suitable file client based on the URI and arguments.

        Args:
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. Default: None.
            uri (str | Path, optional): Uri to be parsed that contains the file
                prefix. Default: None.

        Examples:
            >>> uri = 's3://path/of/your/file'
            >>> file_client = FileClient.infer_client(uri=uri)
            >>> file_client_args = {'backend': 'petrel'}
            >>> file_client = FileClient.infer_client(file_client_args)

        Returns:
            FileClient: Instantiated FileClient object.
        """
        assert file_client_args is not None or uri is not None
        if file_client_args is None:
            file_prefix = cls.parse_uri_prefix(uri)  # type: ignore
            return cls(prefix=file_prefix)
        else:
            return cls(**file_client_args)

    @classmethod
    def _register_backend(cls, name, backend, force=False, prefixes=None):
        if not isinstance(name, str):
            raise TypeError('the backend name should be a string, '
                            f'but got {type(name)}')
        if not inspect.isclass(backend):
            raise TypeError(
                f'backend should be a class but got {type(backend)}')
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f'backend {backend} is not a subclass of BaseStorageBackend')
        if not force and name in cls._backends:
            raise KeyError(
                f'{name} is already registered as a storage backend, '
                'add "force=True" if you want to override it')

        if name in cls._backends and force:
            cls._overridden_backends.add(name)
        cls._backends[name] = backend

        if prefixes is not None:
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            else:
                assert isinstance(prefixes, (list, tuple))
            for prefix in prefixes:
                if prefix not in cls._prefix_to_backends:
                    cls._prefix_to_backends[prefix] = backend
                elif (prefix in cls._prefix_to_backends) and force:
                    cls._overridden_prefixes.add(prefix)
                    cls._prefix_to_backends[prefix] = backend
                else:
                    raise KeyError(
                        f'{prefix} is already registered as a storage backend,'
                        ' add "force=True" if you want to override it')

    @classmethod
    def register_backend(cls, name, backend=None, force=False, prefixes=None):
        """Register a backend to FileClient.

        This method can be used as a normal class method or a decorator.

        .. code-block:: python

            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

            FileClient.register_backend('new', NewBackend)

        or

        .. code-block:: python

            @FileClient.register_backend('new')
            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

        Args:
            name (str): The name of the registered backend.
            backend (class, optional): The backend class to be registered,
                which must be a subclass of :class:`BaseStorageBackend`.
                When this method is used as a decorator, backend is None.
                Defaults to None.
            force (bool, optional): Whether to override the backend if the name
                has already been registered. Defaults to False.
            prefixes (str or list[str] or tuple[str], optional): The prefixes
                of the registered storage backend. Default: None.
                `New in version 1.3.15.`
        """
        if backend is not None:
            cls._register_backend(
                name, backend, force=force, prefixes=prefixes)
            return

        def _register(backend_cls):
            cls._register_backend(
                name, backend_cls, force=force, prefixes=prefixes)
            return backend_cls

        return _register

    def get(self, filepath: Union[str, Path]) -> Union[bytes, memoryview]:
        """Read data from a given ``filepath`` with 'rb' mode.

        Note:
            There are two types of return values for ``get``, one is ``bytes``
            and the other is ``memoryview``. The advantage of using memoryview
            is that you can avoid copying, and if you want to convert it to
            ``bytes``, you can use ``.tobytes()``.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes | memoryview: Expected bytes object or a memory view of the
                bytes object.
        """
        return self.client.get(filepath)

    def get_text(self, filepath: Union[str, Path], encoding='utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        return self.client.get_text(filepath, encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` should create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        self.client.put(obj, filepath)

    def put_text(self, obj: str, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` should create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str, optional): The encoding format used to open the
                `filepath`. Default: 'utf-8'.
        """
        self.client.put_text(obj, filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str, Path): Path to be removed.
        """
        self.client.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return self.client.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
                ``False`` otherwise.
        """
        return self.client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
                otherwise.
        """
        return self.client.isfile(filepath)

    def join_path(self, filepath: Union[str, Path],
                  *filepaths: Union[str, Path]) -> str:
        """Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        return self.client.join_path(filepath, *filepaths)

    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]) -> Iterable[str]:
        """Download data from ``filepath`` and write the data to local path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Note:
            If the ``filepath`` is a local path, just return itself.

        .. warning::
            ``get_local_path`` is an experimental interface that may change in
            the future.

        Args:
            filepath (str or Path): Path to be read data.

        Examples:
            >>> file_client = FileClient(prefix='s3')
            >>> with file_client.get_local_path('s3://bucket/abc.jpg') as path:
            ...     # do something here

        Yields:
            Iterable[str]: Only yield one path.
        """
        with self.client.get_local_path(str(filepath)) as local_path:
            yield local_path

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        yield from self.client.list_dir_or_file(dir_path, list_dir, list_file,
                                                suffix, recursive)
