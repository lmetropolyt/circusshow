import cv2
from cvzone.PoseModule import PoseDetector
from PIL import Image, ImageDraw
import time

detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = detector.findPose(img, draw=True)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True, draw=False)

    for point1 in lmList:
        for point2 in lmList:
            try:
                start_point = point1[0:2]  # (x, y) coordinates of the starting point
                end_point = point2[0:2] # (x, y) coordinates of the ending point
                line_color = (0, 255, 0) # BGR color format (Green in this case)
                line_thickness = 1      # Thickness of the line

                # Draw the line on the frame
                cv2.line(img, start_point, end_point, line_color, line_thickness)
            finally:
                print("", end='')

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    # time.sleep(5)
cap.release()
cv2.destroyAllWindows()
