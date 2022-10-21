from datetime import datetime
from difflib import diff_bytes
from enum import Enum
import cv2
import numpy as np


class Detector:
    # def draw_angled_rec(x0, y0, width, height, angle, img):

    #     _angle = angle * np.pi / 180.0
    #     b = np.cos(_angle) * 0.5
    #     a = np.sin(_angle) * 0.5
    #     pt0 = (int(x0 - a * height - b * width),
    #            int(y0 + b * height - a * width))
    #     pt1 = (int(x0 + a * height - b * width),
    #            int(y0 - b * height - a * width))
    #     pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    #     pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    #     cv2.line(img, pt0, pt1, (255, 255, 255), 3)
    #     cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    #     cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    #     cv2.line(img, pt3, pt0, (255, 255, 255), 3)

    def videoCap(self, cap: cv2.VideoCapture) -> None:
        previous_frame = None  # предыдущий кадр
        prepared_frame = None  # подготовленный кадр

        while True:
            _, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            roi = frame[0: 720,  340: 720]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            prepared_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

            if (previous_frame is None):
                previous_frame = prepared_frame
                continue

            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            # kernel = np.ones((5, 5))
            # diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # thresh_frame = cv2.threshold(diff_frame, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

            contours, h = cv2.findContours(diff_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:

                area = cv2.contourArea(contour)
                if area < 1000 or area > 30000:
                    continue
                x, y, w, h = cv2.boundingRect(contour)

                buttomLeft = [x, y+h]
                buttomRight = [x+w, y+h]

                cv2.circle(roi, buttomLeft, 3, (255, 255, 255), -1)
                cv2.circle(roi, buttomRight, 3, (0, 255, 255), -1)

                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(roi, buttomLeft, (buttomLeft[0] + w, buttomLeft[1]-(h//6)), (255, 0, 0), 2)

            cv2.line(frame, [555, 45], [335, 720], (0, 255, 0), 2)
            cv2.line(frame, [595, 45], [625, 720], (0, 255, 0), 2)

            cv2.imshow("Frame", diff_frame)
            cv2.imshow("roi", roi)
            cv2.imshow("prepared_frame", frame)

            key = cv2.waitKey(30)
            if key == 27:
                # esc key
                break
            if key == 32:
                # space key
                cv2.imwrite("frame.png", frame)
                print('save')

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture('tests.mp4')
    detector = Detector()
    detector.videoCap(cap)
