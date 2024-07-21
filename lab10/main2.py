import numpy as np
import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from pathlib import Path


if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    DIR = Path(sys.argv[0]).parent.resolve()
    # путь для кэширования датасета MNIST
    MNIST = DIR / "mnist"
    MNIST.mkdir(exist_ok=True)

    # загружает датасет MNIST
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist", # мя загружаемого датасета
        data_dir=str(MNIST), # где кэшировать датасет
        # какие части датасета интересуют (split)
        split=["train", "test"],
        # отдельно входные и выходные данные как кортеж
        as_supervised=True,
        with_info=True # етаинформация - описание датасета
    )
    print(ds_info.splits["train"])