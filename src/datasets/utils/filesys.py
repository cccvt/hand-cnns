import os


def get_child_folders(roots):
    child_folders = []
    child_files = []
    for root in roots:
        root, dirnames, filenames = list(os.walk(root))[0]
        path_dirs = [os.path.join(root, dirname) for dirname in dirnames]
        path_files = [os.path.join(root, filename) for filename in filenames]
        child_folders = child_folders + path_dirs
        child_files = child_files + path_files
    return child_folders, child_files


def recursive_files_dataset(root, ext=".jpg", depth=3):
    """
    Returns all the paths to the files with extension ext
    that are less deepely nested then depth

    :param ext: extension of the files to keep
    :param depth: maximum folder depth to explor
    """
    roots = [root]
    file_paths = []
    for level in range(depth):
        child_folders, child_files = get_child_folders(roots)
        roots = child_folders
        file_paths = file_paths + child_files
        if not child_folders:
            break
    file_paths = [
        filename for filename in file_paths if filename.endswith(ext)]
    return file_paths
