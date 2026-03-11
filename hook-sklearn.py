from PyInstaller.utils.hooks import get_package_paths
import os

# Find sklearn base path
sklearn_base, _ = get_package_paths('sklearn')
sklearn_libs = os.path.join(sklearn_base, '.libs')

# Only add DLLs if the folder exists
if os.path.exists(sklearn_libs):
    datas += [
        (os.path.join(sklearn_libs, f), os.path.join('sklearn', '.libs'))
        for f in os.listdir(sklearn_libs)
        if f.endswith('.dll')
    ]