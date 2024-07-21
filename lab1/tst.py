import numpy as np

if __name__ == '__main__':
    arr_d = np.ones((5,5), int)
    R, C = 0, 1
    H, W = 3, 5
    arr_d[R:R+H, C:C+W] = 0
    print(arr_d)
    arr_d.astype()