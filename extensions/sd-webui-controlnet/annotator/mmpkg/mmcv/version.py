# Copyright (c) OpenMMLab. All rights reserved.
__version__ = '1.3.17'


def parse_version_info(version_str: str, length: int = 4) -> tuple:
    """Parse a version string into a tuple.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int | str]: The version info, e.g., "1.3.0" is parsed into
            (1, 3, 0, 0, 0, 0), and "2.0.0rc1" is parsed into
            (2, 0, 0, 0, 'rc', 1) (when length is set to 4).
    """
    from packaging.version import parse
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        release.extend(list(version.pre))
    elif version.is_postrelease:
        release.extend(list(version.post))
    else:
        release.extend([0, 0])
    return tuple(release)


version_info = tuple(int(x) for x in __version__.split('.')[:3])

__all__ = ['__version__', 'version_info', 'parse_version_info']
