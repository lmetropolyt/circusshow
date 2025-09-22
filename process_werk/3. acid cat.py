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

# Function from https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image
    

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    img = detector.findPose(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True, draw=False)

    height, width, channels = img.shape 

    lizzie = segmentor.removeBG(img, (0,0,0))

    cat = cv2.imread('cat.png')
    cat = cv2.resize(cat,(640, 480))
    # cat = resize_with_pad(cat, (640,480), (0,0,0))

    img = img + cat
    img = img + lizzie

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    # time.sleep(5)
cap.release()
cv2.destroyAllWindows()
