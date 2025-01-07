import numpy as np
import cv2
from math import pi, fabs, sin, cos, tan
from typing import Tuple, List


def getHitDetectionMask(image: np.ndarray, ell: Tuple[Tuple[float, float], Tuple[float, float], float]):
    imcpy = image.copy()
    (_,_), (_, _), angle = ell
    imcpy = cv2.medianBlur(imcpy, ksize=5)
    imcpy = cv2.Canny(imcpy, 50,150)
    thr = (image.shape[0]+image.shape[1])/8
    lines = cv2.HoughLines(imcpy, rho=1, theta=pi/180, threshold=int(thr+0.5))
    imline = np.ones(imcpy.shape)

    rang = (image.shape[0]+image.shape[1])/2
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if fabs(angle-(180*theta/pi)%180) < 20:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + rang * (-b))
                y1 = int(y0 + rang * (a))
                x2 = int(x0 - rang * (-b))
                y2 = int(y0 - rang * (a))

                imline = cv2.line(imline.copy(), (x1, y1), (x2, y2), 0, 1)
    imdilate = cv2.erode(imline, np.ones((7,7)))
    return imdilate

def getBinDiff(image: np.ndarray, mask: np.ndarray):
    imcpy = image.copy()

    pixels = imcpy.flatten()
    hist, _ = np.histogram(pixels, bins=256, range=(0,256))
    cdf = np.cumsum(hist)
    cdf_normalized = cdf/cdf[-1]
    threshold_value = np.searchsorted(cdf_normalized, 0.95)

    _, imcpy = cv2.threshold(imcpy, threshold_value, 255, cv2.THRESH_BINARY)

    imcpy = np.multiply(imcpy, mask)

    return imcpy.astype(np.uint8)

def getLines(image: np.ndarray, ell: Tuple[Tuple[float, float], Tuple[float, float], float]):
    (_,_), (_, _), angle = ell
    imcpy = image.copy()

    i = 2
    while True:
        thr = (image.shape[0]+image.shape[1])/i
        lines = cv2.HoughLines(imcpy, rho=1, theta=pi/180, threshold=int(thr+0.5))
        if lines is not None:
            br = False
            for line in lines:
                _, theta = line[0]
                if fabs((180*theta/pi)%180 - angle) < 30:
                    br = True
            if br:
                break
        i+=1

    return lines


class ArrowLine:
    def __init__(self, rho: float, theta: float):
        self.__rho = rho
        self.__thetaRad = theta

        self.__ylinspace = None
        self.__xlinspace = None
        self.__lineTrace = None
        self.__rectSurface = None
        self.__leftEndpoint = np.array([0,0])
        self.__rightEndpoint = np.array([0,0])

    def calculateCoordinates(self, yshape: int, xshape: int):
        y1 = None
        x1 = None
        y2 = None
        x2 = None

        try:
            y1 = self.__rho/sin(self.__thetaRad)+0.5
        except:
            y1 = np.iinfo(np.int16).max

        if y1 < 0:
            y1 = 0
            x1 = int(self.__rho/cos(self.__thetaRad)+0.5)
            x2 = int(x1-tan(self.__thetaRad)*yshape+0.5)
            if x2 > xshape-1:
                x2 = xshape-1
                y2 = int((xshape-1-x1)*tan(self.__thetaRad-pi/2)+0.5)
            else:
                y2 = yshape-1
        elif y1 > yshape-1:
            y1 = yshape-1
            x2 = int(self.__rho/cos(self.__thetaRad)+0.5)
            if x2 > xshape-1:
                y2 = int((x2-yshape-1)/tan(self.__thetaRad)+0.5)
                x1 = int((xshape-1) - x2*(yshape-1-y2)/(yshape-1)+0.5)
                x2 = xshape-1               
            else:
                x1 = int(x2-(yshape-1)*tan(self.__thetaRad)+0.5)
                y2 = 0
        else:
            x1 = 0
            y2 = int(y1 - (xshape-1)/tan(self.__thetaRad)+0.5)
            if y2 < 0:
                x2 = int(y1*tan(self.__thetaRad)+0.5)
                y2 = 0
            elif y2 > yshape-1:
                x2 = int((yshape-1-y1)*tan(pi-self.__thetaRad)+0.5)
                y2 = yshape-1
            else:
                x2 = xshape-1

        num = int(max(abs(x1-x2), abs(y1-y2))+0.5)
            
        self.__xlinspace = np.linspace(x1,x2,num, dtype=np.int16)
        self.__ylinspace = np.linspace(y1,y2,num, dtype=np.int16)

    def calculateLineTrace(self, image: np.ndarray, ell: Tuple[Tuple[float, float], Tuple[float, float], float]):
        (_,_), (_, _), angle = ell
        imcpy = image.copy()
        
        thr = (image.shape[0]+image.shape[1])/8
        lines = cv2.HoughLines(imcpy, rho=1, theta=pi/180, threshold=int(thr+0.5))
        imline = np.ones(imcpy.shape)

        x1, x2, y1, y2 = 0, 0, 0, 0
        rang = (image.shape[0]+image.shape[1])/2
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if fabs(angle-90-(180*theta/pi)%180) < 20:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + rang * (-b))
                    y1 = int(y0 + rang * (a))
                    x2 = int(x0 - rang * (-b))
                    y2 = int(y0 - rang * (a))

                    imline = cv2.line(imline.copy(), (x1, y1), (x2, y2), 0, 1)
        imdilate = cv2.erode(imline, np.ones((3,3)))

        trace = image[self.__ylinspace, self.__xlinspace]
        newtrace = np.zeros(trace.shape)

        rect = False
        maxSurface = 0
        currentSurface = 0
        leftIdx = 0
        currentLeftIdx = 0
        rightIdx = trace.shape[0]-1
        for i in range(trace.shape[0]):
            if imdilate[self.__ylinspace[i],self.__xlinspace[i]]:
                newtrace[i] = np.median(trace[max(0,i-2):min(trace.shape[0],i+3)])
            if newtrace[i]:
                currentSurface += 1
                if not rect:
                    rect = True
                    currentLeftIdx = i
            else:
                if rect:
                    rect = False
                    if currentSurface > maxSurface:
                        maxSurface = currentSurface
                        leftIdx = currentLeftIdx
                        rightIdx = i
                    currentSurface = 0
        if currentSurface > maxSurface:
            maxSurface = currentSurface
                
        self.__lineTrace = newtrace
        self.__rectSurface = maxSurface
        self.__leftEndpoint = np.array([self.__ylinspace[leftIdx], self.__xlinspace[leftIdx]])
        self.__rightEndpoint = np.array([self.__ylinspace[rightIdx], self.__xlinspace[rightIdx]])

        return self.__lineTrace

    def getCoords(self):
        return self.__leftEndpoint, self.__rightEndpoint
    
    def getMaxSurface(self):
        return self.__rectSurface 
    

def getCoordinates(image: np.ndarray, ell: Tuple[Tuple[float, float], Tuple[float, float], float], lines: List):
    imcpy = image.copy()
    (_,_), (_, _), angle = ell
    maxRectSurface = 0
    coords = [0,0], [0,0]
    for i, line in enumerate(lines):
        rho, theta = line[0]
        if angle-60 < (180*theta/pi)%180 < angle+60:
            aline = ArrowLine(rho, theta)
            aline.calculateCoordinates(image.shape[0], image.shape[1])
            aline.calculateLineTrace(imcpy, ell)
            if aline.getMaxSurface() > maxRectSurface:
                maxRectSurface = aline.getMaxSurface()
                coords = aline.getCoords()

    return coords