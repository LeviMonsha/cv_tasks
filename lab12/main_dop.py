import sys
from pathlib import Path
import os

os.environ.setdefault(
    'TF_CPP_MIN_LOG_LEVEL',
    '2'
)
os.environ.setdefault(
    'TFHUB_DOWNLOAD_DIR',
    '.'
)
os.environ.setdefault(
    'TFHUB_DOWNLOAD_PROGRESS',
    '1'
)
import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np
import cv2

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = tfhub.load(hub_handle)
sig = hub_module.signatures['serving_default']
inputs = sig.structured_input_signature[1]
outputs = sig.structured_outputs
print('Inputs')
for name, value in inputs.items():
    print(f'\t{name}: {value}')
print('Outputs')
for name, value in outputs.items():
    print(f'\t{name}: {value}')

def prepare_classes(caltech_dir: Path, mask: str) -> list:
    result = []
    all_items = caltech_dir.glob('*')
    for item in all_items:
        if not item.is_dir() and item.name.endswith(mask):
            result.append(item.name)
    return result


def crop_center(image):
    _n, height, width, _chans = image.shape
    min_n = min(height, width)
    x = (width - min_n) // 2
    y = (height - min_n) // 2
    size = min_n
    image = tf.image.crop_to_bounding_box(
        image,
        y, x,
        size, size
    )
    return image

def load_image(impath, size=(256,256)):
    data = np.fromfile(impath, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = image.astype(np.float32)
    image /= 255.0
    image = image[np.newaxis, ...]
    image = crop_center(image)
    image = tf.image.resize(image, size)
    return image

style_file = input('Укажите путь к файлу стиля (ex: name.jpg): ')
mask = input('Введите маску (ex: jpg): ')
path_dir = Path(sys.argv[0]).parent
image_classes = prepare_classes(path_dir, mask)
content_size = (400, 400)
style_size = (256, 256)

for content in image_classes:
    content_image = load_image(f'{content}', content_size)
    style_image = load_image(style_file, style_size)
    style_image = tf.nn.avg_pool(
        style_image,
        ksize=[3,3],
        strides=[1,1],
        padding='SAME'
    )

    outputs = hub_module(content_image, style_image)
    stylized_image = outputs[0]
    output_directory = Path(sys.argv[0]).parent / 'stylized'
    output_directory.mkdir(exist_ok=True)

    output_path = f'stylized_{content.split(f'.{mask}')[0]}.jpg'
    cv2.imwrite(str(output_path), np.squeeze(stylized_image.numpy()) * 255)
    print(f'Изображение стилизовано и сохранено в {output_path}')