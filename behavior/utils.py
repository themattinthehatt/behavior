import os


def get_subdirs(path):
    """Get all first-level subdirectories in a given path (no recursion).

    Parameters
    ----------
    path : :obj:`str`
        absolute path

    Returns
    -------
    :obj:`list`
        first-level subdirectories in :obj:`path`

    """
    if not os.path.exists(path):
        raise ValueError('%s is not a path' % path)
    try:
        s = next(os.walk(path))[1]
    except StopIteration:
        raise StopIteration('%s does not contain any subdirectories' % path)
    if len(s) == 0:
        raise StopIteration('%s does not contain any subdirectories' % path)
    return s
