import cv2
import numpy as np
from typing import Tuple
from enum import Enum

class ConnectionStatus(Enum):
    OK = 0
    ERROR = 1

def captureVideo(cap: cv2.VideoCapture) -> Tuple[np.ndarray, ConnectionStatus]:
    ret = 0
    frame = np.array([[0]])

    ret, frame = cap.read()

    if not ret:
        return (np.array([[0]]), ConnectionStatus.ERROR)

    return (frame, ConnectionStatus.OK)