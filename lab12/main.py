import sys
from pathlib import Path
import os

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2"
)

os.environ.setdefault(
    "TFHUB_DOWNLOAD_DIR",
    "."
)

os.environ.setdefault(
    "TFHUB_DOWNLOAD_PROGRESS",
    "1"
)

import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np
import cv2

if __name__ == '__main__':
    hub_handle = ("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
    hub_module = tfhub.load(hub_handle)
    sig = hub_module.signatures["serving_default"]
    inputs = sig.structured_input_signature[1]
    outputs = sig.structured_outputs
    print("inputs")
    for name, value in inputs.items():
        print(f"\t{name}: {value}")
    print("outputs")
    for name, value in outputs.items():
        print(f"\t{name}: {value}")