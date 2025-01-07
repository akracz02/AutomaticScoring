import numpy as np
import cv2
from game import TargetType
from enum import Enum
from math import sqrt

from targetDetection import targetDetection
from imageProcessing import getTransformationParameters, reduceImageOfEllipseAndGetNewCenter, getBoundriesAndMask,\
    reduceImageOfEllipse, reduceImageAndRemoveBackground, getTransformedImage, detectHit
from hitPlacement import getHitDetectionMask, getBinDiff, getLines, getCoordinates
from hiDetectionDataPrepFunctions import prepareDataSet

class ConnectionStatus(Enum):
    OK = 0
    ERROR = 1

class ArcheryTargetModel():
    def __init__(self, targetType: TargetType):
        self.__ellipse = None
        self.__newEllipse = None
        self.__rotationMatrix = None
        self.__scalingMaitrix = None
        self.__targetCenter = None
        self.__bnds = None
        self.__reducingMask = None
        self.__hitDetectionMask = None
        self.__knn = None
        
        self.__targetType = targetType

    def detectTarget(self, image: np.ndarray):
        if image is not None:
            self.__ellipse = targetDetection(image, self.__targetType)
            if self.__ellipse is not None:
                imReduced, self.__newEllipse = reduceImageOfEllipseAndGetNewCenter(image, self.__ellipse)
                imTrans, self.__rotationMatrix, self.__scalingMaitrix, self.__targetCenter = getTransformationParameters(imReduced, self.__newEllipse)
                self.__bnds, self.__reducingMask = getBoundriesAndMask(imTrans, self.__targetCenter, self.__newEllipse[1][1])
                self.__hitDetectionMask = getHitDetectionMask(imTrans, self.__newEllipse)
                self.__knn = prepareDataSet()
            return self.__ellipse
        return None
    
    def prepareTransformation(self, image: np.ndarray):
        if self.__ellipse is not None and image is not None and self.__newEllipse is None:
            imReduced, self.__newEllipse = reduceImageOfEllipseAndGetNewCenter(image, self.__ellipse)
            imTrans, self.__rotationMatrix, self.__scalingMaitrix, self.__targetCenter = getTransformationParameters(imReduced, self.__newEllipse)
            self.__bnds, self.__reducingMask = getBoundriesAndMask(imTrans, self.__targetCenter, self.__newEllipse[1][1])
            imTransReduced = reduceImageAndRemoveBackground(imTrans, self.__bnds, self.__reducingMask)
            self.__hitDetectionMask = getHitDetectionMask(imTransReduced, self.__newEllipse)
            self.__knn = prepareDataSet()
            

    def drawEllipse(self, image: np.ndarray):
        if self.__ellipse is not None and image is not None:
            return cv2.ellipse(image, self.__ellipse, (0,255,0), 2)
        
    def createEllipse(self, points):
        if len(points) == 5:
            self.__ellipse = cv2.fitEllipse(np.array(points))
        return self.__ellipse
        
    def drawPoints(self, image: np.ndarray, points):
        if image is not None and points is not None:
            for (x,y) in points:
                image = cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        return image
    
    def resetEllipse(self):
        self.__ellipse = None
        self.__newEllipse = None
        self.__rotationMatrix = None
        self.__scalingMaitrix = None
        self.__targetCenter = None
        self.__bnds = None
        self.__reducingMask = None
        self.__hitDetectionMask = None

    def getTransformedImage(self, image: np.ndarray):
        imReduced = reduceImageOfEllipse(image.copy(), self.__ellipse)
        imTrans = getTransformedImage(imReduced.astype(np.float32), self.__rotationMatrix, self.__scalingMaitrix)
        return reduceImageAndRemoveBackground(imTrans, self.__bnds, self.__reducingMask)
    
    def getHit(self, ppframe: np.ndarray, pframe: np.ndarray, frame: np.ndarray):
        if detectHit(ppframe.copy(), pframe.copy(), frame.copy(), self.__knn):
            framecpy = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)
            pframecpy = cv2.cvtColor(pframe.copy(), cv2.COLOR_RGB2GRAY)

            diff = np.abs(framecpy.astype(np.int16)-pframecpy.astype(np.int16)).astype(np.uint8)
            diffBinary = getBinDiff(diff, self.__hitDetectionMask)
            lines = getLines(diffBinary, self.__newEllipse)
            _, rightEnd = getCoordinates(diffBinary, self.__newEllipse, lines)
            print("diffBinary shape: " + str(diffBinary.shape)+"\nright point: "+str(rightEnd))
            distance = (((rightEnd[0]-diffBinary.shape[0]/2)**2+(rightEnd[1]-diffBinary.shape[1]/2)**2)**0.5)
            print("\ndistance: "+str(distance))
            return sqrt((rightEnd[0]-diffBinary.shape[0]//2)**2+(rightEnd[1]-diffBinary.shape[1]//2)**2)/(max(self.__newEllipse[1])/2)
        return None

    