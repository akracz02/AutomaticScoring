import numpy as np
from typing import Tuple
from time import time
from enum import Enum
from collections.abc import Iterable
from math import ceil

class GameType(Enum):
    TO_SPECIFIED_AMOUNT_OF_SETS = 0
    TO_SPECIFIED_AMOUNT_OF_POINTS = 1

class TargetType(Enum):
    REGULAR_1_10 = 0
    REGULAR_5_10 = 1
    REGULAR_6_10 = 2

class ChangeElementCode(Enum):
    CHANGE_PLAYER_NAME = 0
    CHANGE_HIT_VALUE = 2

class GameState(Enum):
    BREAK = 0
    READY_GO = 1
    ROUND = 2
    GAME_OVER = 3

class Game():
    def __init__(self, noPlayers: int, gameType: GameType,\
                 noSets: int, noPoints: int, noArrows: int, targetType: TargetType,\
                    preparationTime: int, shootTime: int):
        self.__noPlayers = noPlayers
        self.__gameType = gameType
        self.__noSets = noSets
        self.__noPoints = noPoints
        self.__noArrows = noArrows
        self.__targetType = targetType
        self.__preparationTime = preparationTime
        self.__shootTime = shootTime

        self.__players = np.array(['Player'+str(i+1) for i in range(self.__noPlayers)])       
        tableRows = None
        if self.__gameType == GameType.TO_SPECIFIED_AMOUNT_OF_SETS:
            tableRows = self.__noSets
        elif self.__gameType == GameType.TO_SPECIFIED_AMOUNT_OF_POINTS:
            tableRows = int(1+(self.__noPlayers*(self.__noPoints-1)+0.5)//2)

        self.__hitTable = np.array([[[0 for _ in range(self.__noArrows)] for _ in range(self.__noPlayers)] for _ in range(tableRows)])
        self.__scoreTable = np.array([0 for _ in range(self.__noPlayers)])
        self.__sumTable = np.array([[0 for _ in range(self.__noPlayers)] for _ in range(self.__noSets)])
    
        self.__playerToken = 0
        self.__setToken = 0
        self.__arrowToken = 0

        self.__timer = 0
        self.__timeStart = 0
        self.__dist = None

        self.__gameState = GameState.BREAK

    def changeElement(self, code: ChangeElementCode, idx: int | Tuple[int,int,int], elementValue: str):
        if code == ChangeElementCode.CHANGE_PLAYER_NAME and isinstance(idx, int):
            if idx > 0 and idx <= self.__noPlayers:
                self.__players[idx] = elementValue
        if code == ChangeElementCode.CHANGE_HIT_VALUE and isinstance(idx, Iterable):
            if len(idx) == 3:
                if all(isinstance(idx[i], int) for i in range(3)):
                    if 0 < idx[0] <= self.__noSets and 0 < idx[1] <= self.__noPlayers and 0 < idx[2] <= self.__noArrows:
                        presentResult = self.calculateResult(idx[0]+1)
                        self.__hitTable[idx[0],idx[1],idx[2]] = int(elementValue)
                        newResult = self.calculateResult(idx[0]+1)

                        if presentResult != newResult:
                            if len(presentResult) == 1:
                                self.__scoreTable[presentResult[0]] -= 2
                            else:
                                self.__scoreTable[presentResult] -= 1
                            if len(newResult) == 1:
                                self.__scoreTable[newResult[0]] += 2
                            else:
                                self.__scoreTable[newResult] += 1
        return newResult

    def calculateResult(self, setIdx: int = 0):
        if setIdx == 0:
            setIdx = self.__setToken
        arrSums = None
        if self.__noPlayers == 1:
            arrSums = np.squeeze(np.sum(self.__hitTable[setIdx-1,:,:]))
        elif self.__noArrows == 1:
            arrSums = np.squeeze(self.__hitTable[setIdx-1,:,:])
        elif self.__noSets == 1:
            arrSums = np.squeeze(np.sum(self.__hitTable[setIdx-1,:,:]))
        else:
            arrSums = np.squeeze(np.sum(np.squeeze(self.__hitTable[setIdx-1,:,:]), axis=1))
        playerIdx = np.where(arrSums == np.max(arrSums))
        if len(playerIdx) == 1:
            self.__scoreTable[playerIdx[0]] += 2
        else:
            self.__scoreTable[playerIdx[0]] += 1

        return np.array(playerIdx)+1
    
        
    def proceedGame(self):
        if self.__gameState == GameState.BREAK:
            if self.__gameType == GameType.TO_SPECIFIED_AMOUNT_OF_SETS:
                if self.__setToken == self.__noSets:
                    self.__gameState = GameState.GAME_OVER

            if self.__gameType == GameType.TO_SPECIFIED_AMOUNT_OF_POINTS:
                if np.max(self.__scoreTable) >= self.__noPoints:
                    self.__gameState = GameState.GAME_OVER

            if self.__gameState != GameState.GAME_OVER:
                self.__setToken += 1
                self.__playerToken = 0
                self.__arrowToken = 1
                self.__gameState = GameState.READY_GO
                self.__timeStart = time()

        if self.__gameState == GameState.READY_GO:
            self.__timer = time()-self.__timeStart
            if self.__timer >= self.__preparationTime:
                self.__playerToken += 1
                if self.__playerToken > self.__noPlayers:
                    self.__arrowToken += 1
                    if self.__arrowToken > self.__noArrows:
                        self.__gameState = GameState.BREAK
                    else:
                        self.__playerToken = 1
                        self.__gameState = GameState.ROUND
                        self.__timeStart = time()
                else:
                    self.__gameState = GameState.ROUND
                    self.__timeStart = time()


        if self.__gameState == GameState.ROUND:
            self.__timer = time()-self.__timeStart

            if self.__timer >= self.__shootTime or self.__dist is not None:
                if self.__dist is None:
                    self.__dist = 2
                score = self.scoreHit(self.__dist)
                self.__sumTable[self.__setToken-1,self.__playerToken-1] += score
                self.__dist = None
                if self.__arrowToken == self.__noArrows and self.__playerToken == self.__noPlayers:
                    self.__gameState = GameState.BREAK
                    self.calculateResult()
                else:
                    self.__gameState = GameState.READY_GO
                self.__timeStart = time()

        if self.__gameState == GameState.GAME_OVER:
            self.__timer = 0

        return self.__gameState

    def scoreHit(self, dist=None):
        if dist is None:
            dist = self.__dist
        try:
            if self.__targetType == TargetType.REGULAR_1_10:
                dst = 10*(1-dist)
                score = ceil(dst)
                finalScore = score
                if -0.0066 < dst-score < 0 and score != 10:
                    if score > -1:
                        finalScore += 1
                    else:
                        finalScore = 0
                else:
                    if score < 0:
                        finalScore = 0

            if self.__targetType == TargetType.REGULAR_5_10:
                dst = 6*(1-dist)
                score = ceil(dst)
                finalScore = score
                if -0.0111 < dst-score < 0 and score != 6:
                    if score > -1:
                        finalScore += 5
                    else:
                        finalScore = 0
                else:
                    if score > 0:
                        finalScore += 4
                    else:
                        finalScore = 0 
            if self.__targetType == TargetType.REGULAR_6_10:
                dst = 5*(1-dist)
                score = ceil(dst)
                finalScore = score
                if -0.0133 < dst-score < 0 and score != 5:
                    if score > -1:
                        finalScore += 6
                    else:
                        finalScore = 0
                else:
                    if score > 0:
                        finalScore += 5
                    else:
                        finalScore = 0
        except:
            finalScore = 0
        self.__hitTable[self.__setToken-1,self.__playerToken-1,self.__arrowToken-1] = finalScore
        return finalScore

    def getTimer(self):
        if self.__gameState == GameState.BREAK:
            return self.__preparationTime
        if self.__gameState == GameState.READY_GO:
            return max(self.__preparationTime-self.__timer, 0)
        if self.__gameState == GameState.ROUND:
            return max(self.__shootTime-self.__timer, 0)
        if self.__gameState == GameState.GAME_OVER:
            return 0
        
    def getSetWinner(self):
        arrSums = np.squeeze(np.sum(np.squeeze(self.__hitTable[self.__setToken-1,:,:]), axis=1))
        return np.where(arrSums == np.max(arrSums))
    
    def getScoreTable(self):
        return self.__scoreTable

    def getHitTable(self):
        return self.__hitTable
    
    def passHit(self, dist: float):
        if self.__gameState == GameState.ROUND:
            self.__dist = dist

    def getGameState(self):
        return self.__gameState
    
    def getTokens(self):
        return self.__setToken, self.__playerToken, self.__arrowToken
    
    def getSumTable(self):
        return self.__sumTable
    