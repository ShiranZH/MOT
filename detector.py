import torch
import numpy as np
import cv2
from time import time


class ObjectDetection:
    def __init__(self, in_file, out_file):
        self.model = self.load_model()
        self.classes = self.model.names
        self.in_file = in_file
        self.out_file = out_file
        self.cap = cv2.VideoCapture(in_file)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def mouse_detect(self, frame):
        x1, y1, x2, y2 = 0, 0, 0, 0

        def on_mouse(event, x, y, flags, param):
            global point1
            if event == cv2.EVENT_LBUTTONDOWN:
                point1 = (x, y)
                cv2.circle(frame, point1, 10, (255, 0, 0), 5)
                print("x:{}, y:{}".format(x, y))
                x1, y1 = x, y
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                cv2.rectangle(frame, point1, (x, y), (0, 255, 0), 15)
            elif event == cv2.EVENT_LBUTTONUP:
                point2 = (x, y)
                cv2.rectangle(frame, point1, point2, (0, 0, 255), 5)
                x2, y2 = x, y

        cv2.imshow('frame', frame)
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', on_mouse)

        return x1, y1, x2, y2

    def get_video(self):
        return self.cap

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame

    def detect(self, frame):
        labels, cord = self.score_frame(frame)
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        centers = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                if self.class_to_label(labels[i]) == 'sports ball':
                    centers.append(np.array([x1, y1, x2, y2]))

        return centers

    def __call__(self):
        player = self.get_video()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MPEG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time()
            ret, frame = player.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            print(f"Frames Per Second: {fps}")
            out.write(frame)


# detection_ball = ObjectDetection("ball.mp4", "ball_od.avi")
# detection_ball()
#
# detection_multi = ObjectDetection("multiobj.avi", "multi_od.avi")
# detection_multi()

