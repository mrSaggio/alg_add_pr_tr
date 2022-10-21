import cv2
import numpy as np


def videoCap(cap: cv2.VideoCapture) -> None:

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        roi = frame[0: 720,  340: 720]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        cv2.line(frame, [555, 45], [335, 720], (0, 255, 0), 2)
        cv2.line(frame, [595, 45], [625, 720], (0, 255, 0), 2)

        pts1 = np.float32([[555, 45], [595, 45], [335, 720], [625, 720]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 720], [300, 720]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(frame, M, (300, 720))

        _, th1 = cv2.threshold(gray_frame, 230, 255, 0,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("Frame", frame)
        cv2.imshow("roi", roi)
        cv2.imshow("preoprazovanie", dst)
        cv2.imshow("threshold", th1)

        key = cv2.waitKey(30)
        if key == 27:
            # esc key
            break
        if key == 32:
            # space key
            cv2.imwrite('frame.png', frame)
            print('save')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture('1.mp4')
    videoCap(cap)
