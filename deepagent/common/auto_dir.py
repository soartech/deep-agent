import os


def get_numbered_directories(parent_dir, prefix='', suffix=''):
    numbered_directories = []
    for directory in os.listdir(parent_dir):
        try:
            directory = remove_prefix(directory, prefix)
            directory = remove_suffix(directory, suffix)

            integer = int(directory)
            numbered_directories.append(integer)
        except ValueError:
            pass
    return numbered_directories


def get_next_numbered_directory(parent_dir, prefix='', suffix='', make_dirs=True):
    numbered_directories = get_numbered_directories(parent_dir, prefix=prefix, suffix=suffix)

    next_directory = os.path.join(parent_dir, prefix + '1' + suffix) if len(
        numbered_directories) == 0 else os.path.join(
        parent_dir, prefix + str(max(numbered_directories) + 1) + suffix)

    if make_dirs:
        os.makedirs(next_directory)
    return next_directory


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:len(text) - len(suffix)]
    return text
