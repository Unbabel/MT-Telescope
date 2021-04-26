import os
from io import StringIO


def telescope_cache_folder():
    if "HOME" in os.environ:
        cache_directory = os.environ["HOME"] + "/.cache/mt-telescope/"
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
        return cache_directory
    else:
        raise Exception("HOME environment variable is not defined.")


def read_lines(file):
    if file is not None:
        file = StringIO(file.getvalue().decode())
        lines = [line.strip() for line in file.readlines()]
        return lines
    return None
