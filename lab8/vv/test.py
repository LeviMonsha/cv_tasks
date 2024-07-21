import time

for i in range(10):
    print('\x08' * 11, end='')
    print(i, end='', flush=True)
    print("hello", end='', flush=True)
    time.sleep(1)