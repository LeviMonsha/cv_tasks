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

class Clicker:
    """Окно для выбора точек на изображении"""
    def __init__(self, wndname: str, image: np.ndarray):
        """Конструктор экземпляров окна"""
        self._wnd: str = wndname
        self._image: np.ndarray = image
        self._rad: int = 5
        self._color = (255, 0, 255)
        cv2.namedWindow(self._wnd, cv2.WINDOW_AUTOSIZE) # создаем окно
        cv2.setMouseCallback(self._wnd, self._mouse_event)
        self.clicks: tp.List[tp.Tuple[int, int]] = list()

    def _mouse_event(self, event, x, y, flags, param) -> None:
        """Обработчик событий от мыши"""
        if event == cv2.EVENT_LBUTTONUP:
            self.clicks.append((x, y)) # запоминаем выбранную точку
        elif event == cv2.EVENT_RBUTTONUP:
            if self.clicks: # если список не пуст - есть выбранные точки
                del self.clicks[-1] # удаляем последнюю выбранную точку
        else:
            return
        self.update() # перерисуем окно, если произошло событие

        print(event, x, y, flags, param)

    def update(self) -> None:
        """Метод отрисовки содержимого окна"""
        copy = self._image.copy()
        for x, y in self.clicks: # рисуем маркеры
            cv2.circle(copy, (x, y), self._rad, self._color, -1)

        cv2.imshow(self._wnd, copy)

    def close(self) -> None:
        """Метод закрытия окна"""
        cv2.destroyWindow(self._wnd)

    def __enter__(self):
        """вызывается при входе в блок with"""
        self.update()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Вызывается при выходе из блока with"""
        self.close()


if __name__ == '__main__':
    img = load_image("times-square.jpg")
    with Clicker("img", img) as wnd:
        while len(wnd.clicks) < 4:
            key = cv2.waitKey(100)
            if key == 27: # нажат esc
                sys.exit(0) # выход из скрипта
    print(wnd.clicks)

    image_pts = np.array(wnd.clicks, dtype=np.float32)
    # определяем размер изображения
    width = int(max(
        np.linalg.norm(image_pts[0] - image_pts[1]), # длина верхней стороны
        np.linalg.norm(image_pts[2] - image_pts[3]) # длина нижней стороны
    ))
    height = int(max(
        np.linalg.norm(image_pts[0] - image_pts[3]), # длина левой стороны
        np.linalg.norm(image_pts[2] - image_pts[1]) # длина правой стороны
    ))
    # углы изображения-постера
    poster_pts = np.array([
        (0, 0), # левый верхний
        (width - 1, 0), # правый верхний
        (width - 1, height - 1), # правый нижний
        (0, height - 1) # левый нижний
    ], dtype=np.float32)
    # рассчет проективного преобразования
    matrix = cv2.getPerspectiveTransform(
        image_pts, # точки-прообразы ("откуда")
        poster_pts # точки-образы ("куда")
    )
    # преобразуем изображение
    warped = cv2.warpPerspective( # перспективное преобразование
        img, # преобразуемое изображение
        matrix, # матрица преобразования
        (width, height) # желаемый размер целевого изображения
    )
    print(matrix)

    cv2.imshow("warped", warped)
    cv2.waitKey()