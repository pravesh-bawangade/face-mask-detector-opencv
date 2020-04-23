from imutils import face_utils
import dlib
import cv2


def main():
    """
    Face Mask Detection
    :return: None
    """
    detector = dlib.get_frontal_face_detector()
    mouth_cascade = cv2.CascadeClassifier("mouth.xml")
    if mouth_cascade.empty():
        raise IOError('Unable to load the mouth cascade classifier xml file')

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            (x_f, y_f, w_f, h_f) = face_utils.rect_to_bb(rect)
            mouth_rects = mouth_cascade.detectMultiScale(gray[y_f:y_f+h_f, x_f:x_f+w_f], 1.7, 11)

            if len(mouth_rects) == 0:
                cv2.rectangle(image, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 255, 0), 2)
                cv2.putText(image, "Face #{}".format(i + 1), (x_f - 10, y_f - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 0, 255), 4)
                cv2.putText(image, "Face Without Mask #{}".format(i + 1), (x_f - 10, y_f - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4)

        cv2.imshow("Output", image)
        if (cv2.waitKey(5) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

