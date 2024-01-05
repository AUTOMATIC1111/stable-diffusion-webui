from os import scandir
import os.path as path
from typing import Dict, List, Union, Callable, Optional, Iterator
from dataclasses import dataclass, field
from installer import print_dict
from collections import UserDict
import itertools

WasDirty = bool
DidDelete = bool
IsDirectory = bool
DirectoryExists = bool
IsDirectory = bool
IsDirty = bool
CachedDirectoryIsStale = bool
MTime = float
IsHidden = bool

FilePath = str
FilePathList = List[FilePath]
FilePathIterator = Iterator[FilePath]

DirectoryPath = str
DirectoryPathList = List[DirectoryPath]
DirectoryPathIterator = Iterator[DirectoryPath]

class Directory:
    ...
    
DirectoryList = List[Directory]
DirectoryIterator = Iterator[Directory]
DirectoryCollection = Dict[DirectoryPath, Directory]

ExtensionFilter = Callable
ExtensionList = list[str]

RecursiveType = Union[bool,Callable]


def real_path(directory_path:DirectoryPath) -> DirectoryPath | None:
    try:
        return path.abspath(path.expanduser(directory_path))
    except Exception:
        pass
    return None


@dataclass(slots=True,frozen=True)
class Directory(Directory):


    path: DirectoryPath = field(default_factory=str)
    mtime: float = field(default_factory=float, init=False)
    files: FilePathList = field(default_factory=list)
    directories: DirectoryPathList = field(default_factory=list)


    def __post_init__(self):
        object.__setattr__(self, 'mtime', self.live_mtime)


    @classmethod
    def from_dict(cls, dict_object: dict) -> Directory:
        directory = cls.__new__(cls)
        object.__setattr__(directory, 'path', dict_object.get('path'))
        object.__setattr__(directory, 'mtime', dict_object.get('mtime'))
        object.__setattr__(directory, 'files', dict_object.get('files'))
        object.__setattr__(directory, 'directories', dict_object.get('directories'))
        return directory
    

    def clear(self) -> None:
        self._update(Directory.from_dict({
            'path': None,
            'mtime': float(),
            'files': [],
            'directories': []
        }))
    

    def update(self, source_directory: Directory) -> Directory:
        if source_directory is not self:
            self._update(source_directory)
        return self
    
    
    def _update(self, source:Directory) -> None:
        assert not source.path or source.path == self.path, f'When updating a directory, the paths must match.  Attemped to update Directory `{self.path}` with `{source.path}`'
        for dead_path in self.directories:
            if dead_path not in source.directories:
                delete_cached_directory(dead_path)
        self.directories[:] = source.directories
        self.files[:] = source.files
        object.__setattr__(self, 'mtime', source.mtime)
    

    def __str__(self) -> str:
        return str(print_dict(self, path=self.path, mtime=self.mtime, files=len(self.files), directories=len(self.directories)))


    @property
    def exists(self) -> DirectoryExists:
        return self.path and path.exists(self.path)
    

    @property
    def is_directory(self) -> IsDirectory:
        return self.exists and path.isdir(self.path)


    @property
    def live_mtime(self) -> MTime:
        return path.getmtime(self.path) if self.is_directory else 0
    

    @property
    def is_stale(self) -> CachedDirectoryIsStale:
        return not self.is_directory or self.mtime != self.live_mtime


class DirectoryCache(UserDict, DirectoryCollection):
    def __delattr__(self, directory_path: str) -> None:
        directory: Directory = get_directory(directory_path, fetch=False)
        if directory:
            map(delete_cached_directory, directory.directories)
            directory.clear()
        del self.data[directory_path]


def clean_directory(directory: Directory, /, recursive: RecursiveType=False) -> bool:
    if not directory.is_directory:
        is_clean = False
        delete_cached_directory(directory.path)
    else:
        is_clean = not directory.is_stale
        if not is_clean:
            directory.update(fetch_directory(directory.path))
        else:
            for directory_path in directory.directories[:]:
                try:
                    recurse = recursive and (not callable(recursive) or recursive(directory.path))
                    directory = get_directory(directory_path, fetch=recurse)
                    if directory:
                        if directory.is_directory:
                            if recurse:
                                is_clean = clean_directory(directory, recursive=recurse) and is_clean
                            continue
                        delete_cached_directory(directory_path)
                    # If we had intended to fetch this directory, but didn't, that means it doesn't exist. Purge.
                    if recurse:
                        directory.directories.remove(directory_path)
                    is_clean = False
                except Exception:
                    pass
    return is_clean


def get_directory(directory_or_path: DirectoryPath, /, fetch:bool=True) -> Directory | None:
    if isinstance(directory_or_path, Directory):
        if directory_or_path.is_directory:
            return directory_or_path
        else:
            directory_or_path = directory_or_path.path
    global cache_folders
    directory_or_path = real_path(directory_or_path)
    if not cache_folders.get(directory_or_path, None):
        if fetch:
            directory = fetch_directory(directory_path=directory_or_path)
            if directory:
                cache_folders[directory_or_path] = directory
    else:
        clean_directory(cache_folders[directory_or_path])
    return cache_folders[directory_or_path] if directory_or_path in cache_folders else None


def fetch_directory(directory_path: DirectoryPath) -> Directory | None:
    directory: Directory
    for directory in _walk(directory_path, lambda e, path: delete_cached_directory(path), recurse=False):
        return directory
    return None


def _walk(top, onerror:Callable=None, /, recurse:RecursiveType=True) -> Directory:
    # A near-exact copy of `path.walk()`, trimmed slightly. Probably not nessesary for most people's collections, but makes a difference on really large datasets.
    nondirs = []
    walk_dirs = []
    try:
        scandir_it = scandir(top)
    except OSError as error:
        if callable(onerror):
            onerror(error, top)
        return
    with scandir_it:
        while True:
            try:
                try:
                    entry = next(scandir_it)
                except StopIteration:
                    break
            except OSError as error:
                if callable(onerror):
                    onerror(error, top)
                return
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            if not is_dir:
                nondirs.append(entry.path)
            else:
                try:
                    if entry.is_symlink() and not path.exists(entry.path):
                        raise NotADirectoryError('Broken Symlink')
                    walk_dirs.append(entry.path)
                except OSError as error:
                    if callable(onerror):
                        onerror(error, entry.path)
    yield Directory(top, nondirs, walk_dirs)
    if recurse:
        # Recurse into sub-directories
        for new_path in walk_dirs:
            if path.basename(new_path).startswith('models--'):
                continue
            if callable(recurse) and not recurse(new_path):
                continue
            yield from _walk(new_path, onerror, recurse=recurse)


def _cached_walk(top, onerror:Callable=None, /, recurse:RecursiveType=True) -> Directory:
    top = get_directory(top)
    if not top:
        return
    yield top
    if recurse:
        for child_directory in top.directories:
            if path.basename(child_directory).startswith('models--'):
                continue
            if callable(recurse) and not recurse(child_directory):
                continue
            yield from _cached_walk(child_directory, onerror, recurse=recurse)

def walk(top, onerror:Callable=None, /, recurse:RecursiveType=True, cached=True) -> Directory:
    if cached:
        yield from _cached_walk(top, onerror, recurse=recurse)
    else:
        yield from _walk(top, onerror, recurse=recurse)


def delete_cached_directory(directory_path:DirectoryPath) -> DidDelete:
    global cache_folders
    if directory_path in cache_folders:
        del cache_folders[directory_path]


def is_directory(dir_path:DirectoryPath) -> IsDirectory:
    return dir_path and path.exists(dir_path) and path.isdir(dir_path)
    

def directory_mtime(directory_path:DirectoryPath, /, recursive:RecursiveType=True) -> MTime:
    return float(max(0, *[directory.mtime for directory in get_directories(directory_path, recursive=recursive)]))


def unique_directories(directories:DirectoryPathList, /, recursive:RecursiveType=True) -> DirectoryPathIterator:
    '''Ensure no empty, or duplicates'''
    '''If we are going recursive, then directories that are children of other directories are redundant'''
    directories = list(sorted(unique_paths(directories), reverse=True))
    #shared.log.debug(f'Directories: {directories}')
    while directories:
        directory = directories.pop()
        #shared.log.debug(f'yeilding: {directory}')
        yield directory
        if not recursive:
            continue
        _directory = path.join(directory, '')
        while directories and directories[-1].startswith(_directory):
            if not callable(recursive) or not child_directory:
                #shared.log.debug(f'removing `{directories[-1]}` ... {_directory}')
                directories.pop()
                continue
            child_directory = directories[-1][len(directory):]
            #shared.log.debug(f'Checking: {directories[-1]} -> {_directory}')
            if child_directory:
                #shared.log(f'working with {child_directory} -> {directory}')
                next_directory = _directory
                if not callable(recursive):
                    _remove_directory = next_directory
                else:
                    for sub_directory in child_directory.split(path.sep):
                        next_directory = path.join(next_directory, sub_directory)
                        try:
                            if recursive(next_directory):
                                _remove_directory = path.join(next_directory, '')
                                break
                        except Exception:
                            raise # I had thougths about suppressing the excepton, but it's probably better to not.
                while _remove_directory and directories:
                    _d = directories.pop()
                    #shared.log.info(f'Doing the while thing: {_remove_directory} - {_d}')
                    if not directories[-1].startswith(_remove_directory):
                        del _remove_directory


def unique_paths(directory_paths:DirectoryPathList) -> DirectoryPathIterator:
    return (
        key 
        for key 
        in { 
            real_directory_path: True 
            for real_directory_path 
            in filter(bool, [
                real_path(directory_path) 
                for directory_path 
                in filter(bool, directory_paths)
            ])
        }
    )


def get_directories(*directory_paths: DirectoryPathList, fetch:bool=True, recursive:RecursiveType=True) -> DirectoryCollection:
    return filter(
        bool,
        (
            get_directory(directory_path, fetch=fetch) 
            for directory_path in unique_directories(
                directory_paths, recursive=recursive
            )
        )
    )


def directory_files(*directories_or_paths: DirectoryPathList|DirectoryList, recursive: RecursiveType=True) -> FilePathIterator:
    return itertools.chain.from_iterable(
        itertools.chain(
            directory_object.files, 
            []
            if not recursive
            else itertools.chain.from_iterable(
                directory_files(directory, recursive=recursive)
                for directory
                in filter(
                    bool, 
                    map(
                        get_directory, 
                        filter(
                            (
                                ( bool if recursive else False )
                                if not callable(recursive) 
                                else recursive
                            ), 
                            directory_object.directories
                        )
                    )
                )
            )
        )
        for directory_object
        in filter(
            bool,
            map(
                get_directory,
                directories_or_paths
            )
        )
    )


def extension_filter(ext_filter: Optional[ExtensionList]=None, ext_blacklist: Optional[ExtensionList]=None) -> ExtensionFilter:
    if ext_filter:
        ext_filter = [*map(str.upper, ext_filter)]
    if ext_blacklist:
        ext_blacklist = [*map(str.upper, ext_blacklist)]
    def filter_functon(fp:str):
        return (not ext_filter or any(fp.upper().endswith(ew) for ew in ext_filter)) and (not ext_blacklist or not any(fp.upper().endswith(ew) for ew in ext_blacklist))
    return filter_functon


def not_hidden(filepath: FilePath) -> IsHidden:
    return not path.basename(filepath).startswith('.')


def filter_files(file_paths: FilePathList, ext_filter: Optional[ExtensionList]=None, ext_blacklist: Optional[ExtensionList]=None) -> FilePathIterator:
    #print(f'File Paths: {list(file_paths)}')
    return filter(extension_filter(ext_filter, ext_blacklist), file_paths)


def list_files(*directory_paths:DirectoryPathList, ext_filter: Optional[ExtensionList]=None, ext_blacklist: Optional[ExtensionList]=None, recursive:RecursiveType=True) -> FilePathIterator:
    return filter_files(itertools.chain.from_iterable(
        directory_files(directory, recursive=recursive)
        for directory 
        in get_directories(
            *directory_paths, recursive=recursive
        )
    ), ext_filter, ext_blacklist)


cache_folders = DirectoryCache({})