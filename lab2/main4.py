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

N = 5 # размер окрестности
C = 0 # смещение

if __name__ == '__main__':
    img = load_image("sonnet.png")
    gray = cv2.cvtColor( # перобразует изобр. из 1 системы в другую
        img, # изображение, которое преобразуем
        cv2.COLOR_BGR2GRAY # желаемое преобразование
    )

    N, C = map(int, input(f"Введите размер окрестности N и смещение C: ").split())
    if N > 1 or N%2==0:
        N = 3
    if C <= 0:
        C = 1

    while True:
        print(f"окно {N}x{N}, смещение {C}")
        binary = cv2.adaptiveThreshold( # даптивное пороговое преобразование
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C, # простое среднее арифметическое
            cv2.THRESH_BINARY, # простое пороговое преобразование
            N,
            C
        )

        cv2.imshow("gray", gray)
        cv2.imshow("binary", binary)
        key = cv2.waitKeyEx()
        if key == ESC:
            break
        elif key == LEFT:
            if N > 3:
                N -= 2
        elif key == RIGHT:
            N += 2
        elif key == UP:
            C += 5
        elif key == DOWN:
            C -= 5
    cv2.destroyAllWindows()
