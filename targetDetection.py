import numpy as np
from typing import List, Tuple
import cv2
from math import inf, fabs
from statistics import mean
from enum import Enum
from game import TargetType

from imageProcessing import getContours, getEllipsesOfContours, getRed, getBlue

class TargetContoursOlympicTarget(Enum):
    # enum type represents which rings were found in detection process

    InRedInBlue = 0
    InRedOutBlue = 1
    OutRedInBlue = 2
    OutRedOutBlue = 3

def getDetectionMatrix(redEllipses: List, blueEllipses: List, targetType: TargetType):
    # creates matrix which represents quality of a pair of red and blue ellipse

    m, n = len(redEllipses), len(blueEllipses)

    # first criteria: how far from each other are ellipses centres
    centerDistanceAccMatrix = np.zeros((m,n), dtype=float)

    # second criteria: are axes length matching for specific target
    axesRatioMatrix = np.zeros((m,n), dtype=float)

    # third criteria: is ratio between major and minor axis similiar in red and blue axis
    adequateAxesRatioMatrix = np.zeros((m,n), dtype=float)

    # matrix which determines what is estimated placement of the rings
    axesPlacementMatrix = np.zeros((m,n), dtype=TargetContoursOlympicTarget)

    # for every pair of ellipses: distance between centers, ratio of axes and difference between angles is calculated
    for r, redEll in enumerate(redEllipses):
        for b, blueEll in enumerate(blueEllipses):
            (yr, xr), (axmajorR, axminorR), _ = redEll
            (yb, xb), (axmajorB, axminorB), _ = blueEll

            centerDistanceAccMatrix[r,b] = ((xr-xb)**2+(yr-yb)**2)**0.5

            ratioDiff = inf
            ratio = (axmajorB/axmajorR + axminorB/axminorR)/2

            if targetType == TargetType.REGULAR_1_10 or targetType == TargetType.REGULAR_5_10:
                if fabs(ratio-1) < ratioDiff:
                    ratioDiff = fabs(ratio-1)
                    axesRatioMatrix[r,b] = ratioDiff
                    axesPlacementMatrix[r,b] = TargetContoursOlympicTarget.OutRedInBlue
                if fabs(ratio-1.5) < ratioDiff:
                    ratioDiff = fabs(ratio-1.5)
                    axesRatioMatrix[r,b] = ratioDiff
                    axesPlacementMatrix[r,b] = TargetContoursOlympicTarget.OutRedOutBlue
                if fabs(ratio-2) < ratioDiff:
                    ratioDiff = fabs(ratio-2)
                    axesRatioMatrix[r,b] = ratioDiff
                    axesPlacementMatrix[r,b] = TargetContoursOlympicTarget.InRedInBlue
                if fabs(ratio-3) < ratioDiff:
                    ratioDiff = fabs(ratio-3)
                    axesRatioMatrix[r,b] = ratioDiff
                    axesPlacementMatrix[r,b] = TargetContoursOlympicTarget.InRedOutBlue

            elif targetType == TargetType.REGULAR_6_10:
                if fabs(ratio-1) < ratioDiff:
                    ratioDiff = fabs(ratio-1)
                    axesRatioMatrix[r,b] = ratioDiff
                    axesPlacementMatrix[r,b] = TargetContoursOlympicTarget.OutRedInBlue
                if fabs(ratio-1.5) < ratioDiff:
                    ratioDiff = fabs(ratio-1.2)
                    axesRatioMatrix[r,b] = ratioDiff
                    axesPlacementMatrix[r,b] = TargetContoursOlympicTarget.OutRedOutBlue
                if fabs(ratio-2) < ratioDiff:
                    ratioDiff = fabs(ratio-2)
                    axesRatioMatrix[r,b] = ratioDiff
                    axesPlacementMatrix[r,b] = TargetContoursOlympicTarget.InRedInBlue
                if fabs(ratio-3) < ratioDiff:
                    ratioDiff = fabs(ratio-2.5)
                    axesRatioMatrix[r,b] = ratioDiff
                    axesPlacementMatrix[r,b] = TargetContoursOlympicTarget.InRedOutBlue               
            
            adequateAxesRatioMatrix[r,b] = fabs(axmajorR/axminorR-axmajorB/axminorB)

    # every matrix is normalised
    centerDistanceAccMatrix = centerDistanceAccMatrix - np.min(centerDistanceAccMatrix)
    centerDistanceAccMatrix = centerDistanceAccMatrix/np.max(centerDistanceAccMatrix)
    centerDistanceAccMatrix = 1-centerDistanceAccMatrix

    axesRatioMatrix = axesRatioMatrix - np.min(axesRatioMatrix)
    axesRatioMatrix = axesRatioMatrix/np.max(axesRatioMatrix)
    axesRatioMatrix = 1-axesRatioMatrix

    adequateAxesRatioMatrix = adequateAxesRatioMatrix - np.min(adequateAxesRatioMatrix)
    adequateAxesRatioMatrix = adequateAxesRatioMatrix/np.max(adequateAxesRatioMatrix)
    adequateAxesRatioMatrix = 1-adequateAxesRatioMatrix

    # multiplication of each is returned
    return np.multiply(np.multiply(centerDistanceAccMatrix, axesRatioMatrix), adequateAxesRatioMatrix), axesPlacementMatrix

def getBestEllipse(redEllipses, blueEllipses, detectionMatrix, axesPlacementMatrix, targetType: TargetType):
    # creates target from best pair

    # takes the gratest ellipses according to accuracy matrix
    r, b = np.unravel_index(np.argmax(detectionMatrix), detectionMatrix.shape)

    (yr, xr), (axmajorR, axminorR), angleR = redEllipses[r]
    (yb, xb), (axmajorB, axminorB), angleB = blueEllipses[b]

    # center is an average
    y, x = mean([yr,yb]), mean([xr,xb])

    axmajor = None
    axminor = None

    # according to estimated position major and minor axes are created
    if targetType == TargetType.REGULAR_1_10:
        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.InRedInBlue:
            axmajor = (5*axmajorR + 2.5*axmajorB)/2
            axminor = (5*axminorR + 2.5*axminorB)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.InRedOutBlue:
            axmajor = (5*axmajorR + 5*axmajorB/3)/2
            axminor = (5*axminorR + 5*axminorB/3)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.OutRedInBlue:
            axmajor = (2.5*axmajorR + 2.5*axmajorB)/2
            axminor = (2.5*axminorR + 2.5*axminorB)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.OutRedOutBlue:
            axmajor = (2.5*axmajorR + 5*axmajorB/3)/2
            axminor = (2.5*axminorR + 5*axminorB/3)/2

    elif targetType == TargetType.REGULAR_5_10:
        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.InRedInBlue:
            axmajor = (3*axmajorR + 1.5*axmajorB)/2
            axminor = (3*axminorR + 1.5*axminorB)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.InRedOutBlue:
            axmajor = (3*axmajorR + axmajorB)/2
            axminor = (3*axminorR + axminorB)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.OutRedInBlue:
            axmajor = (1.5*axmajorR + 1.5*axmajorB)/2
            axminor = (1.5*axminorR + 1.5*axminorB)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.OutRedOutBlue:
            axmajor = (1.5*axmajorR + axmajorB)/2
            axminor = (1.5*axminorR + axminorB)/2

    elif targetType == TargetType.REGULAR_6_10:
        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.InRedInBlue:
            axmajor = (2.5*axmajorR + 1.25*axmajorB)/2
            axminor = (2.5*axminorR + 1.25*axminorB)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.InRedOutBlue:
            axmajor = (2.5*axmajorR + axmajorB)/2
            axminor = (2.5*axminorR + axminorB)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.OutRedInBlue:
            axmajor = (1.25*axmajorR + 1.25*axmajorB)/2
            axminor = (1.25*axminorR + 1.25*axminorB)/2

        if axesPlacementMatrix[r,b] == TargetContoursOlympicTarget.OutRedOutBlue:
            axmajor = (1.25*axmajorR + axmajorB)/2
            axminor = (1.25*axminorR + axminorB)/2

    # angle is an average
    angle = mean([angleR, angleB])

    return ((y,x), (axmajor, axminor), angle)

def targetDetection(image: np.ndarray, targetType: TargetType):
    RedImage = getRed(image)
    BlueImage = getBlue(image)

    ContoursResult = getContours(RedImage, BlueImage)
    if ContoursResult is None:
        return None

    RedContours, BlueContours = ContoursResult

    AccuracyResult =  getEllipsesOfContours(image, RedContours, BlueContours)
    if AccuracyResult is None:
        return None
    
    redEllipses, blueEllipses = AccuracyResult

    detectionMatrix, axesPlacementMatrix = getDetectionMatrix(redEllipses, blueEllipses, targetType)
        
    return getBestEllipse(redEllipses, blueEllipses, detectionMatrix, axesPlacementMatrix, targetType)