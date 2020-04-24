import numpy as np
import cv2


def main():
    """
    Face Mask Detection
    :return: None
    """

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    mouth_cascade = cv2.CascadeClassifier("mouth.xml")
    if mouth_cascade.empty():
        raise IOError('Unable to load the mouth cascade classifier xml file')

    vs = cv2.VideoCapture(0)
    while True:
        ret, frame = vs.read()
        frame = cv2.resize(frame, (640, 360))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            mouth_rects = mouth_cascade.detectMultiScale(gray[startY:endY, startX:endX], 1.7, 11)

            if len(mouth_rects) == 0:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, "Face #{}".format(i + 1), (startX - 10, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)
                cv2.putText(frame, "Face Without Mask #{}".format(i + 1), (startX - 10, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, "Face Mask Detector", (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 125, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
