import cv2


cap = cv2.VideoCapture("ball.mp4")


od = OrangeDetector()

kf = KalmanFilter()




while True:
    ret, frame = cap.read()
    if ret is False:
        break

    orange_bbox = od.detect(frame)
    x, y, x2, y2 = orange_bbox
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    predicted = kf.predict(cx, cy)
    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), -1)
    cv2.circle(frame, (cx, cy), 20, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


