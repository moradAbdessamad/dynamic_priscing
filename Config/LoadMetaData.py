# Config/LoadMetaData.py

import json
import os

metadata_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '..', 'SRC', 'metaData', 'modelMetaData.json'
)

metadata_path = os.path.normpath(metadata_path)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)
