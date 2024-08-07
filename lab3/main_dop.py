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

def make_mask(shape: tuple, radius: int) -> np.ndarray:
    xr = np.arange(0, shape[1]) - shape[1] // 2
    yr = np.arange(0, shape[0]) - shape[0] // 2
    squares = xr[np.newaxis, :]**2 + yr[:, np.newaxis]**2
    return squares <= radius**2

ESC   = 0x00000018
UP    = 0x00260000
DOWN  = 0x00280000
LEFT  = 0x00250000
RIGHT = 0x00270000
SPACE = 0x00000020

def view_amp(fourier: np.ndarray) -> np.ndarray:
    res = np.abs(fourier)
    res = np.log(res)
    rmin, rmax = res.min(), res.max()
    res -= rmin
    res *= 255.0 / (rmax - rmin)
    return res.astype(np.uint8)

if __name__ == '__main__':
    img = load_image("bridge.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b_mask = True

    radius1 = gray.shape[0] // 8
    radius2 = radius1 // 2

    while(True):
        #mask = make_mask(gray.shape, radius)
        mask1 = make_mask(gray.shape, radius1)
        mask2 = make_mask(gray.shape, radius2)
        mask = mask1 ^ mask2  # mask = mask1 ^ mask2

        print(f"gray {gray.shape[1]}x{gray.shape[0]}, {gray.dtype}:",
              f"{gray.min()}...{gray.max()}")
        cv2.imshow("gray", gray)
        fourier = np.fft.fft2(gray)  # преобразование Фурье
        fourier = np.fft.fftshift(fourier)  # перевод в каноническую форму

        if b_mask: fourier[mask] = 1e-10
        else: fourier[~mask] = 1e-10

        print(f"fourier {fourier.shape[1]}x{fourier.shape[0]}, {fourier.dtype}:",
              f"{fourier.min()}...{fourier.max()}")
        amp = view_amp(fourier)
        print(f"amplitudes {amp.shape[1]}x{amp.shape[0]}, {amp.dtype}:",
              f"{amp.min()}...{amp.max()}")
        cv2.imshow("amp", amp)
        # обратное преобразование
        unshift = np.fft.ifftshift(fourier) # перевод из канонической формы
        restored = np.fft.ifft2(unshift) # обратное преобразование Фурье
        print(f"amplitudes {restored.shape[1]}x{restored.shape[0]}, {restored.dtype}:",
              f"{restored.min()}...{restored.max()}")
        res = np.abs(restored).astype(np.uint8)
        cv2.imshow("res", res)
        key = cv2.waitKeyEx()
        div = gray.shape[0] // 2
        if key == ESC: break
        elif key == UP:
            if radius1 < div:
                radius1 += 1
        elif key == DOWN:
            if radius1 > 0:
                radius1 -= 1
        elif key == LEFT:
            if radius2 > 0:
                radius2 -= 1
        elif key == RIGHT:
            if radius2 < div:
                radius2 += 1
        elif key == SPACE:
            if b_mask: b_mask = False
            else: b_mask = True
    cv2.destroyAllWindows()