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

if __name__ == '__main__':
    img = load_image("lena.png")
    gray = cv2.cvtColor( # перобразует изобр. из 1 системы в другую
        img, # изображение, которое преобразуем
        cv2.COLOR_BGR2GRAY # желаемое преобразование
    )

    gray = gray.clip(max=255-64) + 64
    threshold, binary = cv2.threshold( # пороговое преобразование
        gray,
        0, # значение не используется т.к. используем алг. Оцу
        255, # значение `белых` пикселей
        # желаемый вариант порогового преобразования
        cv2.THRESH_BINARY | cv2.THRESH_OTSU # используем алг. Оцу
    )
    print(f"порог яркости по Оцу: {threshold}")

    cv2.imshow("gray", gray)
    cv2.imshow("binary", binary)
    cv2.waitKey()