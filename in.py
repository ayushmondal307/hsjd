import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('out.avi', fourcc, 20, (640, 480))
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = cam.read()
    if not ret:
        break

bg = np.flip(bg, axis = 1)

while (cam.isOpened()):
    ret, img = cam.read()
    if not ret:
        break
    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 170])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1+mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.unit6))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.unit6))

    mask2 = cv2.bitwise_not(mask1)
    ref1 = cv2.bitwise_and(img, img, mask = mask2)
    ref2 = cv2.bitwise_and(bg, bg, mask = mask1)

    final = cv2.addWeighted(ref1, 1, ref2, 1, 0)

    output.write(final)
    cv2.imshow('magic', final)
    if cv2.waitKey(2) == 32:
       break

cam.release()
cv2.destroyAllWindows()
