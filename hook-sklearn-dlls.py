# hook-sklearn-dlls.py
import os
import sys

# PyInstaller unpacks files to sys._MEIPASS
dist_path = os.path.join(sys._MEIPASS, 'sklearn', '.libs')
if os.path.exists(dist_path):
    os.environ['PATH'] = dist_path + os.pathsep + os.environ['PATH']