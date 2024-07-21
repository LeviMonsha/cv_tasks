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

def crop_center(image):
    _n, height, width, _chans = image.shape
    size = min(height, width)
    x = (width - size) // 2
    y = (height - size) // 2
    image = tf.image.crop_to_bounding_box(
        image,
        y, x,
        size, size
    )
    return image

def load_image(impath, size=(256, 256)):
    data = np.fromfile(impath, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = image.astype(np.float32)
    image /= 255.0
    image = image[np.newaxis, ...]
    image = crop_center(image)
    image = tf.image.resize(image, size)
    return image

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

    content_file = "content.jpg"
    content_size = (400, 400)
    style_file = "style.jpg"
    style_size = (256, 256) # размер стиля рекомендуемый

    content_image = load_image(content_file, content_size)
    style_image = load_image(style_file, style_size)
    style_image = tf.nn.avg_pool(
        style_image,
        ksize=[3, 3],
        strides=[1, 1],
        padding="SAME"
    )

    outputs = hub_module(content_image, style_image)
    stylized_image = outputs[0]

    print(stylized_image.shape, style_image.dtype)

    cv2.imshow("content", np.squeeze(content_image))
    cv2.imshow("style", np.squeeze(style_image))
    cv2.imshow("result", np.squeeze(stylized_image))
    cv2.waitKey()

