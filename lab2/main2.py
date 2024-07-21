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

if __name__ == '__main__':
    img = load_image("lena.png")
    gray = cv2.cvtColor( # перобразует изобр. из 1 системы в другую
        img, # изображение, которое преобразуем
        cv2.COLOR_BGR2GRAY # желаемое преобразование
    )
    threshold = 128

    while True:
        print(f"порог: {threshold}")
        _, binary = cv2.threshold( # пороговое преобразование
            gray,
            threshold, # значение порога яркости
            255, # значение `белых` пикселей
            # желаемый вариант порогового преобразования
            cv2.THRESH_BINARY
        )

        cv2.imshow("gray", gray)
        cv2.imshow("binary", binary)
        key = cv2.waitKeyEx()
        if key == ESC:
            break
        elif key == UP:
            threshold += 5
        elif key == DOWN:
            threshold -= 5
        threshold = min(255, max(0, threshold))
    cv2.destroyAllWindows()
