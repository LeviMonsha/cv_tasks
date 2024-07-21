import cv2

if __name__ == '__main__':
    img = cv2.imread("lena.png")

    if img is None:
        print("изображение загрузить не удалось")
    else:
        h, w, ch = img.shape
        print("изображение загрузилось")
        print(f"изображение с каналами {w}x{h} с {ch} каналами")