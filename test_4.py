import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture('video.mp4')
cv2.namedWindow("TrackBars")
cv2.createTrackbar("L-H", "TrackBars", 0, 180, nothing)
cv2.createTrackbar("L-S", "TrackBars", 0, 255, nothing)
cv2.createTrackbar("L-V", "TrackBars", 0, 255, nothing)
cv2.createTrackbar("U-H", "TrackBars", 180, 180, nothing)
cv2.createTrackbar("U-S", "TrackBars", 255, 255, nothing)
cv2.createTrackbar("U-V", "TrackBars", 255, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (960, 540))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "TrackBars")
    l_s = cv2.getTrackbarPos("L-S", "TrackBars")
    l_v = cv2.getTrackbarPos("L-V", "TrackBars")
    u_h = cv2.getTrackbarPos("U-H", "TrackBars")
    u_s = cv2.getTrackbarPos("U-S", "TrackBars")
    u_v = cv2.getTrackbarPos("U-V", "TrackBars")

    lower_red = np.array([85, 30, 0])
    upper_red = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel)
    # mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        aprox = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x = aprox.ravel()[0]
        y = aprox.ravel()[1]
        print(cnt,"\n")
        if 500 < area < 70000:

            cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 5)
            if len(aprox) == 3:
                cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
            elif len(aprox) == 4:
                cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 0, 0))
            elif 10 < len(aprox) < 20:
                cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0))

    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    cv2.imshow('Mask', mask)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
