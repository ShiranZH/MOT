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

# With these parameters, the tracker works the most perfectly.
kf0 = KalmanFilter(10, 2, 161, 1, 0.01, 0.01)
kf1 = KalmanFilter(10, 621, 152, 1, 0.001, 0.001)
kfs = [kf0, kf1]


while True:
    ret, frame = cap.read()
    if ret is False:
        break

    centers = od.detect(frame)
    for idx, center in enumerate(centers):
        if idx >= 2:
            continue
        x1, y1, x2, y2 = center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        print(cx, cy)
        # Draw
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Get index
        (kx0, ky0) = kf0.getxy()
        (kx1, ky1) = kf1.getxy()
        print(kx0, ky0, kx1, ky1)
        if abs(kx0-cx)+abs(ky0-cy) < abs(kx1-cx)+abs(ky1-cy):
            kf = kf0
        else:
            kf = kf1

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
        cv2.circle(frame, (int(x[0,0]), int(y[0,0])), 3, (0, 255, 0), -1)
        cv2.circle(frame, (int(nx[0,0]), int(ny[0,0])), 3, (0, 0, 255), -1)

        out.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break


