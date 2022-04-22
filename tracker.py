import cv2
import numpy as np
from detector import ObjectDetection
from kalmanfilter import KalmanFilter


# in_file = "ball.mp4"
# out_file = "ball_track.avi"
in_file = "multiobj.avi"
out_file = "multi_track.avi"

od = ObjectDetection(in_file, out_file)
cap = od.get_video()

x_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
y_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
four_cc = cv2.VideoWriter_fourcc(*"MPEG")
out = cv2.VideoWriter(out_file, four_cc, 20, (x_shape, y_shape))

# ret, frame = cap.read()
# centers = od.detect(frame)
# x1, y1, x2, y2 = centers[0]
# a1, b1, a2, b2 = centers[1]
# cx = int((x1 + x2) / 2)
# cy = int((y1 + y2) / 2)
# ca = int((a1 + a2) / 2)
# cb = int((b1 + b2) / 2)
# print(cx, cy, ca, cb)
kf0 = KalmanFilter(0.1, 2, 161, 1, 0.1, 0.1)
kf1 = KalmanFilter(0.1, 152, 621, 1, 0.1, 0.1)
kfs = [kf0, kf1]

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    centers = od.detect(frame)
    for idx, center in enumerate(centers):
        if idx >= 2:
            continue
        kf = kfs[idx]
        x1, y1, x2, y2 = center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        print(cx, cy)
        # Draw
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Predict
        (x, y) = kf.predict()
        print(x, y)

        # Update
        cxy = np.array([[cx], [cy]])
        (nx, ny) = kf.update(cxy)
        print(nx, ny)
        cx = int(nx[0,0])
        cy = int(ny[0,0])

        # Draw
        cv2.circle(frame, (int(x[0,0]), int(y[0,0])), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(nx[0,0]), int(ny[0,0])), 10, (0, 0, 255), -1)

        out.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break


