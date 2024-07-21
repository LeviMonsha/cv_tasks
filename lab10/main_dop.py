import numpy as np
import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import sys
from pathlib import Path


if __name__ == '__main__':
    DIR = Path(sys.argv[0]).parent.resolve()
    MNIST = DIR / "mnist.npz"
    # скачивает датасет MNIST
    dataset = tf.keras.datasets.mnist.load_data(
        str(MNIST)
    )
    (train_images, train_labels), (test_images, test_labels) = dataset

    print(f"train images: {train_images.shape} {train_images.dtype}")
    print(f"/t{train_images.min()} {train_images.max()}")
    print(f"train label: {train_labels.shape} {train_labels.dtype}")
    print(f"/t{train_labels.min()} {train_labels.max()}")

    # нормализуем датасет
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # формируем структуру нейронной сети
    model = tf.keras.Sequential( # сеть без обратных связей     # доп задание изменить Sequential
        [ # список слоев или подмоделей
            tf.keras.layers.Input(shape=(28, 28, 1)),
            # входной свёрточный слой из 32 нейронов с окном 3х3 и функцией активации ReLU, форма входных данных (28,28,1)
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            # слой прореживания с окном 2х2
            tf.keras.layers.MaxPooling2D((2, 2)),
            # входной свёрточный слой из 32 нейронов с окном 3х3 и функцией активации ReLU
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            # слой прореживания с окном 2х2
            tf.keras.layers.MaxPooling2D((2, 2)),
            # входной свёрточный слой из 32 нейронов с окном 3х3 и функцией активации ReLU
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),

            # вытягиваем матрицу в одномерный массив
            tf.keras.layers.Flatten(),
            # скрытый слой для вычислений (плотный слой) - 32 нейрона с ReLU
            tf.keras.layers.Dense(64, activation="relu"),
            # выходной слой - логиты
            # нейронов столько же сколько классов объектов
            tf.keras.layers.Dense(10)
        ]
    )

    model.compile(
        # оптимизатор - алгоритм обучения сети
        optimizer="adam", # указываем имя или объект алгоритма
        # функция потерь сети - насколько плохо работает сеть
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True # преобразуем массив выходов в номер макс. выхода
        ),
        # метрики качества работы сети
        metrics=["accuracy"] # точность - доля успехов
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.001, patience=3)

    # обучение модели с механизмом ранней остановки
    model.fit(
        train_images, # входные данные
        train_labels, # "правильные ответы"
        epochs=20, # сколько итераций обучения сети выполнить
        verbose=3,
        callbacks=[early_stopping]
    )
    # проверка качества работы сети
    loss, acc = model.evaluate(
        test_images, # данные для проверки
        test_labels, # ответы для проверки
        verbose=2
    )
    print(f"точность классификации: {acc:.1%}")
    # сохраняем обученную сеть
    model.save(str(DIR / "digits_dop.h5"))