import inspect
import PyInstaller.utils.hooks
print('members', [name for name in dir(PyInstaller.utils.hooks) if 'Tree' in name])
# try import Tree
try:
    from PyInstaller.utils.hooks import Tree
    print('Tree imported')
except Exception as e:
    print('import error', e)
