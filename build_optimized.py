#!/usr/bin/env python3
"""
Optimized build script for AI Text Detector
Reduces executable size by excluding CUDA libraries and using CPU-only PyTorch
"""

import os
import subprocess
import sys
from pathlib import Path

def clean_build():
    """Clean previous build artifacts"""
    print("Cleaning previous builds...")
    os.system('rmdir /s /q build dist 2>nul')

def build_app():
    """Build the application with optimized settings"""

    print("Building AI Text Detector...")

    # Use the optimized spec file
    cmd = [
        'pyinstaller',
        '--clean',
        'main.spec'
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        if exe_path.exists():
            size_gb = exe_path.stat().st_size / (1024**3)
            print(f"Build successful! Executable size: {size_gb:.2f} GB")
            print(f"Executable location: {exe_path.absolute()}")
        else:
            print("Build completed but executable not found")
    else:
        print("Build failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)

def main():
    clean_build()
    build_app()

if __name__ == '__main__':
    main()