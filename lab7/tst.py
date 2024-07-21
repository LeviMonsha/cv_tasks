import numpy as np
import cv2
import time
from collections import Counter

class SVMClicker:
    # метки классов
    LABEL_LIST = [0, 1, 2, 3, 4]
    # цвета точек
    LIST_POINT_COLOR = [(128, 255, 128), (255, 128, 128), (255, 255, 255), (128, 128, 255), (128, 255, 255)]
    # цвета полей классов
    LIST_FIELD_COLOR = [(0, 128, 0), (128, 0, 0), (128, 128, 128), (0, 0, 128), (0, 128, 128)]

    cur_lb = LABEL_LIST[0]

    # # метки классов
    # LABEL_GREEN = +1
    # LABEL_BLUE = -1
    # # цвета точек
    # POINT_GREEN = (128, 255, 128)
    # POINT_BLUE = (255, 128, 128)
    # # цвета полей классов
    # FIELD_GREEN = (0, 128, 0)
    # FIELD_BLUE = (128, 0, 0)

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
                0.1
             )
        )

    def _click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.points.append((x, y, self.cur_lb))
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

        if len(cnt) >= 2 and min(cnt.values()) >= 2:
            points = np.array(self.points, np.float32)
            try:
                self.svm.trainAuto( # обучение классификатора
                    points[:, 0:2], # описание объектов
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
                    if label == self.LABEL_LIST[0]:
                        self._image[r:r+d, c:c+d] = self.LIST_FIELD_COLOR[0]
                    elif label == self.LABEL_LIST[1]:
                        self._image[r:r+d, c:c+d] = self.LIST_FIELD_COLOR[1]
                    elif label == self.LABEL_LIST[2]:
                        self._image[r:r+d, c:c+d] = self.LIST_FIELD_COLOR[2]
                    elif label == self.LABEL_LIST[3]:
                        self._image[r:r+d, c:c+d] = self.LIST_FIELD_COLOR[3]
                    elif label == self.LABEL_LIST[4]:
                        self._image[r:r+d, c:c+d] = self.LIST_FIELD_COLOR[4]


        t3 = time.time()
        print(f"обучение: {(t2 - t1) * 1000:.1f} мс")
        print(f"отрисовка: {(t3 - t2) * 1000:.1f} мс")

        for x, y, lbl in self.points:
            color = self.LIST_POINT_COLOR[lbl]
            # color = self.POINT_GREEN if lbl == self.LABEL_GREEN else self.POINT_BLUE
            cv2.circle(self._image, (x, y), 5, color, -1)

        supvectors = self.svm.getUncompressedSupportVectors()
        if supvectors is not None:
            for x, y in supvectors.astype(np.int32):
                cv2.circle(self._image, (x, y), 8, (0, 0, 0), 2)

        cv2.imshow(self._name, self._image)

if __name__ == '__main__':
    wnd = SVMClicker("SVM", (640, 480))

    wnd.update()
    while True:
        key = cv2.waitKey()
        if key == 27:
            break
        while True:
            key = cv2.waitKey()
            if key == 27:
                break
            elif key == ord("1"):
                wnd.cur_lb = wnd.LABEL_LIST[0]
            elif key == ord("2"):
                wnd.cur_lb = wnd.LABEL_LIST[1]
            elif key == ord("3"):
                wnd.cur_lb = wnd.LABEL_LIST[2]
            elif key == ord("4"):
                wnd.cur_lb = wnd.LABEL_LIST[3]
            elif key == ord("5"):
                wnd.cur_lb = wnd.LABEL_LIST[4]

    if wnd.svm.isTrained():
        wnd.svm.save("svm_nl.xml")