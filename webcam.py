import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation  
from cvzone.PoseModule import PoseDetector
from PIL import Image, ImageDraw
import time
import random
import numpy as np
from typing import Tuple


cap = cv2.VideoCapture(0)
segmentor = SelfiSegmentation(1)

detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

def background_foreground(background, foreground):
    # Overlay two images without messing with colours
    # taken from here https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    return background

def ensure_alpha_channel(img):
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = detector.findPose(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True, draw=False)

    # get lizzie but transparent
    mask_bgr = segmentor.removeBG(img, (0,0,0))
    lizzie = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2BGRA)
    black = np.all(lizzie[:, :, :3] == [0,0,0], axis=-1)
    lizzie[black, 3] = 0

    # # Load cat with alpha channel
    # cat = cv2.imread('cat.png', cv2.IMREAD_UNCHANGED)  
    # cat = cv2.resize(cat, (640, 480))

    img = ensure_alpha_channel(img)
    lizzie = ensure_alpha_channel(lizzie)

    # img = background_foreground(img, cat)
    b, g, r, a = cv2.split(img)
    gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
    img = cv2.merge([gray, gray, gray, a])

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

    img = background_foreground(img, lizzie)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

