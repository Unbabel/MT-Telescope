import os


def telescope_cache_folder():
    if "HOME" in os.environ:
        cache_directory = os.environ["HOME"] + "/.cache/mt-telescope/"
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
        return cache_directory
    else:
        raise Exception("HOME environment variable is not defined.")