import sys
import typing as tp

import numpy as np
import cv2

#~~~~~~~~~~~~~~~~~
def load_image(fpath: str) -> np.ndarray:
    data = np.fromfile(fpath, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("неверный формат файла")
    return img
#~~~~~~~~~~~~~~~~~

ESC   = 0x00000018
UP    = 0x00260000
DOWN  = 0x00280000
LEFT  = 0x00250000
RIGHT = 0x00270000
SPACE = 0x00000020

if __name__ == '__main__':
    img = load_image("sils.png")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    files = [f"sil{i}.png" for i in range(1, 6)]
    color_def = (33, 33, 33)
    colors = [col * color_def for col in range(1, 6)]

    for f, c in zip(files, colors):
        # попарный перебор кортежа
        print(colors)

    ght = cv2.createGeneralizedHoughGuil()
    ght.setLevels(360)
    ght.setMaxBufferSize(1000)
    ght.setXi(45)
    # параметры определения позиции
    ght.setMinDist(50) # минимальное расстояние между фигурами
    ght.setDp(2) # шаг поиска позиций
    ght.setPosThresh(150)

    # араметры определения угла
    ght.setMaxAngle(360)
    ght.setAngleStep(5)
    ght.setAngleEpsilon(5)
    ght.setAngleThresh(50)
    # параметры определения масштаба
    ght.setMinScale(1.0)
    ght.setMaxScale(1.01)
    ght.setScaleStep(0.1)
    ght.setScaleThresh(50)

    # ght.setTemplate(gray_sample)  # фигура, которую ищем
    #
    # copy = img.copy()
    # # выполняем поиск
    # positions, votes = ght.detect(gray_image)
    # if positions is not None:
    #     R = max(gray_sample.shape) // 2
    #     color = (64, 255, 64)
    #     for row in positions[0, :]:
    #         x, y = row[0:2].astype(np.int32)
    #         scale = row[2]
    #         angle = int(row[3]) # в градусах
    #         c = np.cos(np.radians(angle - 90))
    #         s = np.sin(np.radians(angle - 90))
    #         r = int(R * scale)
    #         cv2.circle(
    #             copy,
    #             (x, y), R, color, 2
    #         )
    #         cv2.line(
    #             copy,
    #             (x, y),
    #             (x + int(r * c), y + int(r * s)),
    #             color, 2
    #         )
    # cv2.imshow("res", copy)
    # cv2.waitKey()

