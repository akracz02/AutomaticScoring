import numpy as np
from typing import Tuple, List
import cv2
from math import fabs, sin, cos, pi
from sklearn.neighbors import KNeighborsClassifier

def getRed(image: np.ndarray):
    # This and following use HSV colormap to seperate target colours
    imcopy = image.copy()
    imcopy = cv2.cvtColor(imcopy, cv2.COLOR_RGB2HSV)
    imcopy = np.bitwise_and(np.bitwise_and(np.bitwise_or(imcopy[:,:,0] > 160, imcopy[:,:,0] < 10), imcopy[:,:,1] > 50), imcopy[:,:,2] > 50, dtype=np.uint8)
    for _ in range(np.max(imcopy.shape)//100):
        imcopy = cv2.medianBlur(imcopy, 5)
    return 255*imcopy

def getBlue(image: np.ndarray):
    imcopy = image.copy()
    imcopy = cv2.cvtColor(imcopy, cv2.COLOR_RGB2HSV)
    imcopy = np.bitwise_and(np.bitwise_and(np.bitwise_and(imcopy[:,:,0] > 90, imcopy[:,:,0] < 140), imcopy[:,:,1] > 75), imcopy[:,:,2] > 75, dtype=np.uint8)
    for _ in range(np.max(imcopy.shape)//100):
        imcopy = cv2.medianBlur(imcopy, 5)
    return 255*imcopy

def getYellow(image: np.ndarray):
    imcopy = image.copy()
    imcopy = cv2.cvtColor(imcopy, cv2.COLOR_RGB2HSV)
    imcopy = np.bitwise_and(np.bitwise_and(np.bitwise_and(imcopy[:,:,0] > 20, imcopy[:,:,0] < 40), imcopy[:,:,1] > 50), imcopy[:,:,2] > 50, dtype=np.uint8)
    for _ in range(np.max(imcopy.shape)//100):
        imcopy = cv2.medianBlur(imcopy, 5)
    return 255*imcopy

def getBlack(image: np.ndarray):
    imcopy = image.copy()
    imcopy = cv2.cvtColor(imcopy, cv2.COLOR_RGB2HSV)
    imcopy = np.array(imcopy[:,:,2] < 50, dtype=np.uint8)
    for _ in range(np.max(imcopy.shape)//100):
        imcopy = cv2.medianBlur(imcopy, 5)
    return 255*imcopy

def getWhite(image: np.ndarray):
    imcopy = image.copy()
    imcopy = cv2.cvtColor(imcopy, cv2.COLOR_RGB2HSV)
    imcopy = np.bitwise_and(imcopy[:,:,1] < 30, imcopy[:,:,2] > 200, dtype=np.uint8)
    for _ in range(np.max(imcopy.shape)//100):
        imcopy = cv2.medianBlur(imcopy, 5)
    return 255*imcopy

def getContours(RedBoosted: np.ndarray, BlueBoosted: np.ndarray):
    # function gets contours of an image

    RedBlurred = RedBoosted.copy()
    BlueBlurred = BlueBoosted.copy()

    # multiple median blur
    for _ in range(np.max(RedBoosted.shape)//100):
        RedBlurred = cv2.medianBlur(RedBlurred, 5)
        BlueBlurred = cv2.medianBlur(BlueBlurred, 5)

    # edge detection
    RedEdges = cv2.Canny(RedBlurred,50,150)
    BlueEdges = cv2.Canny(BlueBlurred,50,150)

    # finding contours 
    RedContours, _ = cv2.findContours(RedEdges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    BlueContours, _ = cv2.findContours(BlueEdges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    RedContours = list(RedContours)
    BlueContours = list(BlueContours)

    # if contours not found
    if len(RedContours) == 0 or len(BlueContours) == 0:
        return None

    # fullfilling two list to equal shape (required in specific target detection functions)
    diff = len(RedContours) - len(BlueContours)
    
    for _ in range(abs(diff)):
        if diff > 0:
            BlueContours.append(None)
        else:
            RedContours.append(None)

    return RedContours, BlueContours

def getEllipsesOfContours(image: np.ndarray, RedContours: list, BlueContours: list) -> Tuple[List, List]:
    # function creates accuracy matrix which will be used in determining the best two ellipses

    redEllipses = []
    blueEllipses = []

    # forbidding too small ellipses - reduces matrix shape
    ContourLowLimit = pi*0.01*np.sqrt(image.shape[0]**2+image.shape[1]**2)

    # check every pair of contours
    for redcont, bluecont in zip(RedContours, BlueContours):
        if redcont is not None:
            redcont = np.squeeze(redcont)
            if redcont.shape[0] >= ContourLowLimit:

                # fitting ellipse
                ell = cv2.fitEllipse(redcont.astype(np.int32))
                (ycent, xcent), (axmajor, axminor), angleDeg = ell
                angleRad = pi*angleDeg/180

                # if ellipse is outside the image - it will not be considered
                sizeCorrect = True

                hmajor = fabs(sin(angleRad)*axmajor/2)
                hminor = fabs(sin(angleRad+pi/4)*axminor/2)
                lmajor = fabs(cos(angleRad)*axmajor/2)
                lminor = fabs(cos(angleRad+pi/4)*axminor/2)
                
                if ycent + 2.5*max(hmajor, hminor) > image.shape[0] or ycent - 2.5*max(hmajor, hminor) < 0:
                    sizeCorrect = False
                if xcent + 2.5*max(lmajor, lminor) > image.shape[1] or xcent - 2.5*max(lmajor, lminor) < 0:
                    sizeCorrect = False

                if sizeCorrect:
                    redEllipses.append(ell)

        # same as above
        if bluecont is not None:
            bluecont = np.squeeze(bluecont)
            if bluecont.shape[0] >= 1.5*ContourLowLimit:
                ell = cv2.fitEllipse(bluecont.astype(np.int32))
                (ycent, xcent), (axmajor, axminor), angleDeg = ell
                angleRad = pi*angleDeg/180

                sizeCorrect = True

                hmajor = fabs(sin(angleRad)*axmajor/2)
                hminor = fabs(sin(angleRad+pi/4)*axminor/2)
                lmajor = fabs(cos(angleRad)*axmajor/2)
                lminor = fabs(cos(angleRad+pi/4)*axminor/2)
                
                if ycent + 5*max(hmajor, hminor)/3 > image.shape[0] or ycent - 5*max(hmajor, hminor)/3 < 0:
                    sizeCorrect = False
                if xcent + 5*max(lmajor, lminor)/3 > image.shape[1] or xcent - 5*max(lmajor, lminor)/3 < 0:
                    sizeCorrect = False

                if sizeCorrect:
                    blueEllipses.append(ell)
    
    if len(redEllipses) == 0 or len(blueEllipses) == 0:
        return None

    return redEllipses, blueEllipses


def getTransformationParameters(image: np.ndarray, ellipse: Tuple[Tuple[float,float], Tuple[float, float], float]):
    (y, x), (axmajor, axminor), angle = ellipse
    impng = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    impng[:,:,:3] = image.copy()
    impng[int(x+0.5),int(y+0.5),3] = 255
    rotationMatrix = cv2.getRotationMatrix2D(np.array([y,x]), angle, 1.0)
    rotatedImage = cv2.warpAffine(impng, rotationMatrix, (image.shape[1], image.shape[0]))

    k = max(axminor, axmajor)/min(axminor, axmajor)
    scalingMatrix = np.array([[k, 0, (1-k)*x], [0, 1, 0]])
    scaledImage = cv2.warpAffine(rotatedImage, scalingMatrix, (image.shape[1], image.shape[0]))

    cords = np.squeeze(np.where(scaledImage[:,:,3] != 0))
    cordX, cordY = int(np.median(cords[0,:])+0.5), int(np.median(cords[1,:])+0.5)
    return scaledImage[:,:,:3].astype(np.uint8), rotationMatrix, scalingMatrix, np.array([cordX, cordY])


def getTransformedImage(image: np.ndarray, rotationMatrix: np.ndarray, scalingMatrix: np.ndarray) -> np.ndarray:
    rotatedImage = cv2.warpAffine(image, rotationMatrix.astype(np.float32), (image.shape[1], image.shape[0]))

    return cv2.warpAffine(rotatedImage, scalingMatrix.astype(np.float32), (image.shape[1], image.shape[0])).astype(np.uint8)

def reduceImageOfEllipse(image: np.ndarray, ell: Tuple[Tuple[float, float], Tuple[float, float], float]):
    (x, y), (axmajor, _), _ = ell

    ytop = max(int(y - axmajor+0.5),0)
    ybottom = min(int(y + axmajor+0.5),image.shape[0]-1)
    xleft = max(int(x - axmajor+0.5),0)
    xright = min(int(x + axmajor+0.5),image.shape[1]-1)

    if ytop == 0:
        ybottom = int(2*y+0.5)
    elif ybottom == image.shape[0]-1:
        ytop = int(ybottom - 2*fabs(ybottom-y) + 0.5)
    if xleft == 0:
        xright = int(2*x+0.5)
    elif xright == image.shape[1]-1:
        xleft = int(xright - 2*fabs(xright-x) + 0.5)

    return image[ytop:ybottom, xleft:xright]

def reduceImageOfEllipseAndGetNewCenter(image: np.ndarray, ell: Tuple[Tuple[float, float], Tuple[float, float], float]):
    (x, y), (axmajor, axminor), angle = ell

    ytop = max(int(y - axmajor+0.5),0)
    ybottom = min(int(y + axmajor+0.5),image.shape[0]-1)
    xleft = max(int(x - axmajor+0.5),0)
    xright = min(int(x + axmajor+0.5),image.shape[1]-1)

    if ytop == 0:
        ybottom = int(2*y+0.5)
    elif ybottom == image.shape[0]-1:
        ytop = int(ybottom - 2*fabs(ybottom-y) + 0.5)
    if xleft == 0:
        xright = int(2*x+0.5)
    elif xright == image.shape[1]-1:
        xleft = int(xright - 2*fabs(xright-x) + 0.5)

    newim = image[ytop:ybottom, xleft:xright]
    

    return newim, (((newim.shape[0]+0.5)//2, (newim.shape[1]+0.5)//2), (axmajor, axminor), angle)

def getBoundriesAndMask(image: np.ndarray, centre: Tuple[float, float], radius: float):
    y, x = centre
    ytop = max(int(y - 1.2*radius//2+0.5),0)
    ybottom = min(int(y + 1.2*radius//2+0.5),image.shape[0]-1)
    xleft = max(int(x - 1.2*radius//2+0.5),0)
    xright = min(int(x + 1.2*radius//2+0.5),image.shape[1]-1)

    if ytop == 0:
        ybottom = int(2*y+0.5)
    elif ybottom == image.shape[0]-1:
        ytop = int(ybottom - 2*fabs(ybottom-y) + 0.5)
    if xleft == 0:
        xright = int(2*x+0.5)
    elif xright == image.shape[1]-1:
        xleft = int(xright - 2*fabs(xright-x) + 0.5)

    m, n = ybottom-ytop, xright-xleft

    mask = np.zeros((m,n,3), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            if ((i-m//2)**2 + (j-n//2)**2)**0.5 < 1.2*radius:
                mask[i,j,:] = 1

    return (ytop,ybottom,xleft,xright), mask

def reduceImageAndRemoveBackground(image: np.ndarray, bounds: Tuple[int,int,int,int], mask: np.ndarray):
    imreduced = image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
    return np.multiply(imreduced, mask)

def detectHit(ppframe: np.ndarray, pframe: np.ndarray, frame: np.ndarray, knn: KNeighborsClassifier):
    framecpy = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)
    pframecpy = cv2.cvtColor(pframe.copy(), cv2.COLOR_RGB2GRAY)
    ppframecpy = cv2.cvtColor(ppframe.copy(), cv2.COLOR_RGB2GRAY)

    frame_pframe_diff = np.abs((framecpy.astype(np.int16) - pframecpy.astype(np.int16))).astype(np.uint8)
    pframe_ppframe_diff = np.abs((pframecpy.astype(np.int16)-ppframecpy.astype(np.int16))).astype(np.uint8)

    fpf_std = np.std(frame_pframe_diff)
    pfppf_std = np.std(pframe_ppframe_diff)

    fpf_max = np.max(frame_pframe_diff)
    pfppf_max = np.max(pframe_ppframe_diff)

    pfppf_pred = knn.predict(np.array([[pfppf_std, pfppf_max]]))
    fpf_pred = knn.predict(np.array([[fpf_std, fpf_max]]))
    
    return True if pfppf_pred == 0 and fpf_pred == 1 else False