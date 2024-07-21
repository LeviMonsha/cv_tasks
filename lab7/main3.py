import numpy as np
import cv2
import time
from collections import Counter

class SVMClicker:
    # метки классов
    LABEL_GREEN = +1
    LABEL_BLUE = -1
    # цвета точек
    POINT_GREEN = (128, 255, 128)
    POINT_BLUE = (255, 128, 128)
    # цвета полей классов
    FIELD_GREEN = (0, 128, 0)
    FIELD_BLUE = (128, 0, 0)

    def __init__(self, name: str, size: tuple[int, int]):
        self._name = name
        self._image = np.zeros((size[1], size[0], 3), np.uint8)
        self.points: list[tuple[int, int, int]] = []
        cv2.namedWindow(self._name)
        cv2.setMouseCallback(self._name, self._click)

        # создаем классификатор
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC) # задача классификации
        self.svm.setKernel(cv2.ml.SVM_RBF) # нелинейный вариант
        self.svm.setTermCriteria(
            (
                cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                100, # не более 100 итераций
                0.1 #
             )
        )

    def _click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.points.append((x, y, self.LABEL_GREEN))
        elif event == cv2.EVENT_RBUTTONUP:
            self.points.append((x, y, self.LABEL_BLUE))
        elif event == cv2.EVENT_MBUTTONUP:
            self.points.clear()
        else:
            return
        self.update()

    def update(self):
        self._image.fill(0)
        # проверяем, достаточно ли данных для обучения
        labels = [p[2] for p in self.points]
        cnt = Counter(labels)

        t1 = time.time()

        # если есть оба класса и в обоих есть хотя бы 2 точки
        if len(cnt) == 2 and min(cnt.values()) >= 2:
            points = np.array(self.points, np.float32)
            try:
                self.svm.trainAuto( # обучение классификатора
                    points[:, 0:2], # писание объектов
                    cv2.ml.ROW_SAMPLE, # объекты идут по строкам матрицы
                    points[:, 2].astype(np.int32) # метки
                )
            except Exception as err:
                print("ошибка обучения", err)

        t2 = time.time()

        if self.svm.isTrained(): # обучен ли
            # проверяем работу классификатора
            d = 3 # размер ячейки
            for r in range(0, self._image.shape[0], d):
                for c in range(0, self._image.shape[1], d):
                    # координаты центра
                    x = c + d // 2
                    y = r + d // 2
                    pt = np.array([[x, y]], np.float32)
                    _strength, label = self.svm.predict(pt)
                    if label == self.LABEL_GREEN:
                        self._image[r:r+d, c:c+d] = self.FIELD_GREEN
                    else:
                        self._image[r:r + d, c:c + d] = self.FIELD_BLUE

        t3 = time.time()
        print(f"обучение: {(t2 - t1) * 1000:.1f} мс")
        print(f"отрисовка: {(t3 - t2) * 1000:.1f} мс")

        for x, y, lbl in self.points:
            color = self.POINT_GREEN if lbl == self.LABEL_GREEN else self.POINT_BLUE
            cv2.circle(self._image, (x, y), 5, color, -1)

        supvectors = self.svm.getUncompressedSupportVectors()
        if supvectors is not None:
            for x, y in supvectors.astype(np.int32):
                cv2.circle(self._image, (x, y), 8, (0, 0, 0), 2)

        cv2.imshow(self._name, self._image)

if __name__ == '__main__':
    wnd = SVMClicker("SVM", (640, 480))

    try:
        wnd.svm = wnd.svm.load("svm_nl.xml")
    except Exception as err:
        print("ошибка загрузки")

    wnd.update()
    while True:
        key = cv2.waitKey()
        if key == 27:
            break

    if wnd.svm.isTrained():
        wnd.svm.save("svm_nl.xml")