import cv2

def cut_video():
    # Открываем видеофайл
    video = cv2.VideoCapture('video_cur.mp4')

    # Проверяем, успешно ли открылся файл
    if not video.isOpened():
        print("Error opening video file")

    # Число кадров в секунду
    fps = video.get(cv2.CAP_PROP_FPS)

    # Цикл для обработки каждого кадра видео
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Сохраняем кадр в файл
        cv2.imwrite('negatives/neg%d.jpg' % video.get(cv2.CAP_PROP_POS_FRAMES), frame)

        # # Показываем кадр
        # cv2.imshow('Frame', frame)
        #
        # # Выход из цикла при нажатии клавиши 'q'
        # if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        #     break

    # Очищаем ресурсы
    video.release()
    cv2.destroyAllWindows()

cut_video()