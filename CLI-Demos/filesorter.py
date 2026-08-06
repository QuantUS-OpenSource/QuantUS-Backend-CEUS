# -*- coding: utf-8 -*-
"""
Created on Sat May 27 21:01:22 2023

@author: Yuanshan Wu
"""

import os
import shutil

path = '/Users/emilyqi/Desktop/GRS CEUS PT 2'
despath = '/Users/emilyqi/Desktop/GRS CEUS PT 2/sorted data'

KNOWN_EXTENSIONS = sorted([
    '.mp4',
    '.raw.bmode', '.raw.event', '.raw.nlc', '.raw.xml',
], key=len, reverse=True)

for file_ in os.listdir(path):
    # Skip directories and macOS metadata files (._*)
    if file_.startswith('.') or os.path.isdir(os.path.join(path, file_)):
        continue

    prefix = file_
    for ext in KNOWN_EXTENSIONS:
        if file_.endswith(ext):
            prefix = file_[:-len(ext)]
            break
    else:
        prefix = os.path.splitext(file_)[0]

    folder_path = os.path.join(despath, prefix)
    os.makedirs(folder_path, exist_ok=True)

    shutil.move(os.path.join(path, file_), folder_path)