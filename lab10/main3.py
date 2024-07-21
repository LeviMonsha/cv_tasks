import numpy as np
import cv2

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

class DrawingWindow:
    def __init__(self, name: str, size: int):
        self.name = name
        self.image = np.zeros((size, size), np.uint8)
        self.w = int(2 * size / 28)
        self._prev = None
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self._mouse)
        self.update()

    def update(self):
        cv2.imshow(self.name, self.image)

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONUP:
            self.image.fill(0)
            self._prev = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._prev = (x, y)
        elif event == cv2. EVENT_LBUTTONUP:
            self._prev = None
        elif event == cv2. EVENT_MOUSEMOVE and self._prev:
            cv2.line(self.image, self._prev, (x, y), (255, ), self.w)
            self._prev = (x, y)
        else:
            return
        self.update()

    def get(self) -> np.ndarray:
        output = cv2.resize( # изменяем размер
            self.image, # какое изображение меняем
            (28, 28), # новый размер
            interpolation=cv2.INTER_AREA
        )
        output = output.reshape((1, 28, 28))
        output = output / 255.0
        return output


if __name__ == '__main__':
    trained_model = tf.keras.models.load_model("digits.h5") # файл из доп. задания
    probability_model = tf.keras.Sequential(
        [
            trained_model, # под-модель - компонент
            tf.keras.layers.Softmax() # логиты в вероятности
        ]
    )

    wnd = DrawingWindow("draw", 400)
    while True:
        key = cv2.waitKey()
        if key == 27:
            break
        elif key == 32: # пробел
            inp = wnd.get()
            result = probability_model.predict(inp)
            answer = result[0].argmax()
            probability = result[0, answer]
            print(f"{answer} - {probability:.1%}")