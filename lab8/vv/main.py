import numpy as np
import cv2

class KMeansClicker:
    COLOR = [(255,255,64),
             (255,64,255),
             (64,255,255),
             (255,64,64),
             (64,255,64),
             (255,255,64),
             (64,64,255),
             (64,64,64),
             (255,128,64)
    ]

    def __init__(self, name: str, size: tuple[int, int]):
        self._name = name
        self._image = np.zeros(
            (size[1], size[0], 3),
            np.uint8
        )
        self._points: list[tuple[int, int]] = []
        cv2.namedWindow(self._name)
        cv2.setMouseCallback(self._name, self._click)

    def _click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self._points.append((x, y))
        elif event == cv2.EVENT_RBUTTONUP and self._points:
            del self._points[-1]
        elif event == cv2.EVENT_MBUTTONUP:
            self._points.clear()
        else:
            return
        self.update()

    def update(self):
        self._image.fill(0)
        for x, y in self._points:
            cv2.circle(self._image, (x, y), 3, (255, 255, 255), 1)
        cv2.imshow(self._name, self._image)

    def clusterize(self, k: int):
        if len(self._points) < k:
            return
        criteria  = (
            # останов  по числу итерации и по погрешности
            cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
            10, # не более 10 итераций
            1.0 # пока сдвиг больше
        )
        # начальные позиции центра выбираются случайно
        flags = cv2.KMEANS_RANDOM_CENTERS
        # сколько попыток делать
        attempts = 10
        # подготавливаем данные
        pts = np.array(self._points, np.float32)
        # кластеризация
        _compactness, labels, centers = cv2.kmeans(
            data=pts,
            K=k,
            bestLabels=None,
            criteria=criteria,
            attempts=attempts,
            flags=flags,
            centers=None
        )
        self._image.fill(0)
        # рисуем точки цветами их кластеров
        for p, lbl in zip(self._points, labels[:, 0]):
            color = self.COLOR[int(lbl)]
            cv2.circle(self._image, p, 3, color, -1)
        # рисуем центры кластеров
        for p, color in zip(centers, self.COLOR):
            x, y = p.astype(np.int32)
            cv2.circle(self._image, (x,y), 6, color, 2)
        cv2.imshow(self._name, self._image)

keys = {
    ord(str(i)): i
    for i in range(2, 10)
}
wnd = KMeansClicker('KMeans', (640, 480))
wnd.update()
while True:
    key = cv2.waitKey()
    if key in (-1, 27):
        break
    elif key in keys:
        wnd.clusterize(keys[key])