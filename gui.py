from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QTableWidgetItem
from PyQt5.QtGui import QIntValidator
import pandas as pd
import cv2
import numpy as np

from cameraConnection import ConnectionStatus, captureVideo

from game import Game
from game import TargetType, GameType, GameState
from ArcheryTargetModel import ArcheryTargetModel
from imageProcessing import getRed

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # language preference

        self.__language_box = pd.read_excel("language.xlsx")
        self.__language = self.__language_box.columns[0]

        # used fonts

        self.__font20 = QtGui.QFont()
        self.__font20.setPointSize(20)
        self.__font20.setBold(True)
        self.__font20.setWeight(75)

        self.__font15 = QtGui.QFont()
        self.__font15.setPointSize(15)
        self.__font15.setBold(True)
        self.__font15.setWeight(75)

        self.__font10 = QtGui.QFont()
        self.__font10.setPointSize(10)
        self.__font10.setBold(True)
        self.__font10.setWeight(75)

        # supported targets
        
        self.__TargetPhotosPaths = ['images/REGULAR_1_10.png', 'images/REGULAR_5_10.png', 'images/REGULAR_6_10.png']
        self.__targetType = TargetType.REGULAR_1_10

        # video capturing and target detecting varaibles

        self.__Video = None
        self.__detect = False
        self.__manualMark = False
        self.__markedPoints = []
        self.__backupEllipse = None
        self.__ellipse = None
        self.__connected = False
        self.__disconnect = False

        # image processing frames
        self.__actFrame = None
        self.__prevFrame = None
        self.__prevPrevFrame = None

        # menu widnow index

        self.__WindowIndex = 0

        # line edit validators

        self.__PosIntValidator = QIntValidator(1, 99, self)
        self.__IPAddrValidator = QIntValidator(0, 255, self)
        self.__DroidCamPortValidator = QIntValidator(0, 9999, self)

        # game setings
        self.__noPlayers = None
        self.__gameType = GameType.TO_SPECIFIED_AMOUNT_OF_SETS
        self.__noPoints = None
        self.__preparationTime = None
        self.__shootTime = None
        self.__noSets = None
        self.__noArrows = None

        # game and target controllers
        self.__game = None
        self.__model = None
        self.ScoreTable = None
        self.__proceed = False
        self.__gameState = GameState.BREAK

        self.__BackLayout = QVBoxLayout()

    def setupUi(self, MainWindow):

        # 1. general setup

        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1024, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.MainImage = QtWidgets.QLabel(self.centralwidget)
        self.MainImage.setGeometry(QtCore.QRect(-200, -60, 800, 600))
        self.MainImage.setText("")
        self.MainImage.setPixmap(QtGui.QPixmap("images/MainWindowBackground.jpg"))
        self.MainImage.setObjectName("MainImage")
        self.MainImage.lower()

        self.__BackLayout.addWidget(self.MainImage)

        self.centralwidget.setLayout(self.__BackLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 941, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # 1.1 language setup

        self.LanguageComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.LanguageComboBox.setGeometry(QtCore.QRect(10, 10, 80, 30))
        self.LanguageComboBox.setFont(self.__font10)
        self.LanguageComboBox.setObjectName("LanguageComboBox")
        for _ in range(len(self.__language_box.columns)):
            self.LanguageComboBox.addItem("")
        
        self.LanguageComboBox.currentIndexChanged.connect(self.__changeLanguage)

        # Next and Back push buttons

        self.NextPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.NextPushButton.setGeometry(QtCore.QRect(830, 610, 150, 60))
        self.NextPushButton.setFont(self.__font15)
        self.NextPushButton.setObjectName("Next1PushButton")
        self.NextPushButton.clicked.connect(self.__NextWindow)

        self.BackPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.BackPushButton.setGeometry(QtCore.QRect(500, 610, 150, 60))
        self.BackPushButton.setFont(self.__font15)
        self.BackPushButton.setObjectName("BacktPushButton")
        self.BackPushButton.clicked.connect(self.__PrevWindow)
        self.BackPushButton.hide()

        # 2. first menu window:

        # 2.1 text labels

        self.NoPlayersText = QtWidgets.QLabel(self.centralwidget)
        self.NoPlayersText.setGeometry(QtCore.QRect(500, 20, 400, 40))
        self.NoPlayersText.setFont(self.__font15)
        self.NoPlayersText.setObjectName("NoPlayersText")

        self.NoSetsText = QtWidgets.QLabel(self.centralwidget)
        self.NoSetsText.setGeometry(QtCore.QRect(500, 90, 400, 40))
        self.NoSetsText.setFont(self.__font15)
        self.NoSetsText.setObjectName("NoSetsText")

        self.NoPointsText = QtWidgets.QLabel(self.centralwidget)
        self.NoPointsText.setGeometry(QtCore.QRect(500, 160, 400, 40))
        self.NoPointsText.setFont(self.__font15)
        self.NoPointsText.setObjectName("NoPointsText")

        self.NoArrowsText = QtWidgets.QLabel(self.centralwidget)
        self.NoArrowsText.setGeometry(QtCore.QRect(500, 230, 400, 40))
        self.NoArrowsText.setFont(self.__font15)
        self.NoArrowsText.setObjectName("NoArrowsText")

        self.PreparationTimeText = QtWidgets.QLabel(self.centralwidget)
        self.PreparationTimeText.setGeometry(QtCore.QRect(500, 300, 400, 40))
        self.PreparationTimeText.setFont(self.__font15)
        self.PreparationTimeText.setObjectName("PreparationTimeText")

        self.ShootTimeText = QtWidgets.QLabel(self.centralwidget)
        self.ShootTimeText.setGeometry(QtCore.QRect(500, 370, 400, 40))
        self.ShootTimeText.setFont(self.__font15)
        self.ShootTimeText.setObjectName("ShootTimeText")

        self.GameTypeText = QtWidgets.QLabel(self.centralwidget)
        self.GameTypeText.setGeometry(QtCore.QRect(500, 440, 400, 40))
        self.GameTypeText.setFont(self.__font15)
        self.GameTypeText.setObjectName("GameTypeText")

        # 2.2 line edits

        self.NoPlayersLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.NoPlayersLineEdit.setGeometry(QtCore.QRect(950, 20, 40, 30))
        self.NoPlayersLineEdit.setFont(self.__font15)
        self.NoPlayersLineEdit.setObjectName("NoPlayersLineEdit")
        self.NoPlayersLineEdit.setValidator(self.__PosIntValidator)

        self.NoSetsLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.NoSetsLineEdit.setGeometry(QtCore.QRect(950, 90, 40, 30))
        self.NoSetsLineEdit.setFont(self.__font15)
        self.NoSetsLineEdit.setObjectName("NoSetsLineEdit")
        self.NoSetsLineEdit.setValidator(self.__PosIntValidator)

        self.NoPointsLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.NoPointsLineEdit.setGeometry(QtCore.QRect(950, 160, 40, 30))
        self.NoPointsLineEdit.setFont(self.__font15)
        self.NoPointsLineEdit.setObjectName("NoSetsLineEdit")
        self.NoPointsLineEdit.setValidator(self.__PosIntValidator)

        self.NoArrowsLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.NoArrowsLineEdit.setGeometry(QtCore.QRect(950, 230, 40, 30))
        self.NoArrowsLineEdit.setFont(self.__font15)
        self.NoArrowsLineEdit.setObjectName("NoArrowsLineEdit")
        self.NoArrowsLineEdit.setValidator(self.__PosIntValidator)

        self.PreparationTimeLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.PreparationTimeLineEdit.setGeometry(QtCore.QRect(950, 300, 40, 30))
        self.PreparationTimeLineEdit.setFont(self.__font15)
        self.PreparationTimeLineEdit.setObjectName("PreparationTimeLineEdit")
        self.PreparationTimeLineEdit.setValidator(self.__PosIntValidator)

        self.ShootTimeLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.ShootTimeLineEdit.setGeometry(QtCore.QRect(950, 370, 40, 30))
        self.ShootTimeLineEdit.setFont(self.__font15)
        self.ShootTimeLineEdit.setObjectName("ShootTimeLineEdit")
        self.ShootTimeLineEdit.setValidator(self.__PosIntValidator)

        # combo box

        self.GameTypeComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.GameTypeComboBox.setGeometry(QtCore.QRect(800, 440, 200, 30))
        self.GameTypeComboBox.setFont(self.__font10)
        self.GameTypeComboBox.setObjectName("TargetComboBox")
        self.GameTypeComboBox.addItem("")
        self.GameTypeComboBox.addItem("")
        self.GameTypeComboBox.currentIndexChanged.connect(self.__changeGameType)

    
        # 3. second menu window:

        # 3.1 text labels

        self.PickTargetText = QtWidgets.QLabel(self.centralwidget)
        self.PickTargetText.setGeometry(QtCore.QRect(500, 20, 400, 40))
        self.PickTargetText.setFont(self.__font20)
        self.PickTargetText.setObjectName("PickTargetText")
        self.PickTargetText.hide()

        # 3.3 pixmap

        self.TargetImage = QtWidgets.QLabel(self.centralwidget)
        self.TargetImage.setGeometry(QtCore.QRect(590, 100, 300, 300))
        self.TargetImage.setText("")
        self.TargetImage.setScaledContents(True)
        self.TargetImage.setPixmap(QtGui.QPixmap(self.__TargetPhotosPaths[0]))
        self.TargetImage.setObjectName("TargetImage")
        self.TargetImage.raise_()
        self.TargetImage.hide()

        # 3.4 combobox

        self.TargetComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.TargetComboBox.setGeometry(QtCore.QRect(800, 20, 200, 30))
        self.TargetComboBox.setFont(self.__font10)
        self.TargetComboBox.setObjectName("TargetComboBox")
        for _ in self.__TargetPhotosPaths:
            self.TargetComboBox.addItem("")
        self.TargetComboBox.currentIndexChanged.connect(self.__changeTarget)
        self.TargetComboBox.hide()

        # 4. third menu window

        # 4.1 text lables

        self.IPAddrText = QtWidgets.QLabel(self.centralwidget)
        self.IPAddrText.setGeometry(QtCore.QRect(500, 20, 200, 40))
        self.IPAddrText.setFont(self.__font15)
        self.IPAddrText.setObjectName("IPAddrText")
        self.IPAddrText.hide()

        self.DroidCamPortText = QtWidgets.QLabel(self.centralwidget)
        self.DroidCamPortText.setGeometry(QtCore.QRect(500, 120, 200, 40))
        self.DroidCamPortText.setFont(self.__font15)
        self.DroidCamPortText.setObjectName("DroidCamPortText")
        self.DroidCamPortText.hide()

        # 4.2 line edits

        self.IPLineEdit1 = QtWidgets.QLineEdit(self.centralwidget)
        self.IPLineEdit1.setGeometry(QtCore.QRect(750, 20, 50, 30))
        self.IPLineEdit1.setFont(self.__font15)
        self.IPLineEdit1.setObjectName("IPLineEdit1")
        self.IPLineEdit1.setValidator(self.__IPAddrValidator)
        self.IPLineEdit1.hide()

        self.IPLineEdit2 = QtWidgets.QLineEdit(self.centralwidget)
        self.IPLineEdit2.setGeometry(QtCore.QRect(810, 20, 50, 30))
        self.IPLineEdit2.setFont(self.__font15)
        self.IPLineEdit2.setObjectName("IPLineEdit2")
        self.IPLineEdit2.setValidator(self.__IPAddrValidator)
        self.IPLineEdit2.hide()

        self.IPLineEdit3 = QtWidgets.QLineEdit(self.centralwidget)
        self.IPLineEdit3.setGeometry(QtCore.QRect(870, 20, 50, 30))
        self.IPLineEdit3.setFont(self.__font15)
        self.IPLineEdit3.setObjectName("IPLineEdit3")
        self.IPLineEdit3.setValidator(self.__IPAddrValidator)
        self.IPLineEdit3.hide()

        self.IPLineEdit4 = QtWidgets.QLineEdit(self.centralwidget)
        self.IPLineEdit4.setGeometry(QtCore.QRect(930, 20, 50, 30))
        self.IPLineEdit4.setFont(self.__font15)
        self.IPLineEdit4.setObjectName("IPLineEdit4")
        self.IPLineEdit4.setValidator(self.__IPAddrValidator)
        self.IPLineEdit4.hide()

        self.DroidCamPortLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.DroidCamPortLineEdit.setGeometry(QtCore.QRect(900, 120, 80, 30))
        self.DroidCamPortLineEdit.setFont(self.__font15)
        self.DroidCamPortLineEdit.setObjectName("DroidCamPortLineEdit")
        self.DroidCamPortLineEdit.setValidator(self.__DroidCamPortValidator)
        self.DroidCamPortLineEdit.hide()

        # 4.3 push buttons

        self.ConnectPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.ConnectPushButton.setGeometry(QtCore.QRect(640, 220, 200, 60))
        self.ConnectPushButton.setFont(self.__font15)
        self.ConnectPushButton.setObjectName("ConnectPushButton")
        self.ConnectPushButton.clicked.connect(self.__connectToCamera)
        self.ConnectPushButton.hide()

        self.DisconnectPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.DisconnectPushButton.setGeometry(QtCore.QRect(640, 220, 200, 60))
        self.DisconnectPushButton.setFont(self.__font15)
        self.DisconnectPushButton.setObjectName("DisonnectPushButton")
        self.DisconnectPushButton.clicked.connect(self.__disconnectCamera)
        self.DisconnectPushButton.hide()

        self.DetectPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.DetectPushButton.setGeometry(QtCore.QRect(640, 320, 200, 60))
        self.DetectPushButton.setFont(self.__font15)
        self.DetectPushButton.setObjectName("DetectPushButton")
        self.DetectPushButton.clicked.connect(self.__detectTarget)
        self.DetectPushButton.hide()

        self.ManualMarkPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.ManualMarkPushButton.setGeometry(QtCore.QRect(640, 420, 200, 60))
        self.ManualMarkPushButton.setFont(self.__font15)
        self.ManualMarkPushButton.setObjectName("ManualMarkPushButton")
        self.ManualMarkPushButton.clicked.connect(self.__markTarget)
        self.ManualMarkPushButton.hide()

        self.ResetTargetPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.ResetTargetPushButton.setGeometry(QtCore.QRect(640, 520, 200, 60))
        self.ResetTargetPushButton.setFont(self.__font15)
        self.ResetTargetPushButton.setObjectName("ResetTargetPushButton")
        self.ResetTargetPushButton.clicked.connect(self.__resetTarget)
        self.ResetTargetPushButton.hide()     

        # 5. fourth window

        # 5.1 Push Buttons
        
        self.NextSetPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.NextSetPushButton.setGeometry(QtCore.QRect(640, 420, 150, 60))
        self.NextSetPushButton.setFont(self.__font15)
        self.NextSetPushButton.setObjectName("NextPushButton")
        self.NextSetPushButton.clicked.connect(self.__proceedGame)
        self.NextSetPushButton.hide()

        # 5.2 pixmap

        self.TimeDisplaySquare = QtWidgets.QLabel(self.centralwidget)
        self.TimeDisplaySquare.setGeometry(QtCore.QRect(590, 100, 50, 50))
        self.TimeDisplaySquare.setText("")
        self.TimeDisplaySquare.setScaledContents(True)
        self.TimeDisplaySquare.setPixmap(QtGui.QPixmap('images/red_background.jpg'))
        self.TimeDisplaySquare.setObjectName("TimeDisplaySquare")
        self.TimeDisplaySquare.raise_()
        self.TimeDisplaySquare.hide()

        # 5.3 text labels

        self.TimeDisplayText = QtWidgets.QLabel(self.centralwidget)
        self.TimeDisplayText.setGeometry(QtCore.QRect(590, 100, 50, 50))
        self.TimeDisplayText.setFont(self.__font15)
        self.TimeDisplayText.setObjectName("TimeDisplayText")
        self.TimeDisplayText.hide()

        self.WinnerDisplayText = QtWidgets.QLabel(self.centralwidget)
        self.WinnerDisplayText.setGeometry(QtCore.QRect(50, 50, 600, 50))
        self.WinnerDisplayText.setFont(self.__font15)
        self.WinnerDisplayText.setObjectName("WinnerDisplayText")
        self.WinnerDisplayText.hide()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AutomaticScoring"))
        self.NoPlayersText.setText(_translate("MainWindow", self.__language_box[self.__language][0]))
        self.NoSetsText.setText(_translate("MainWindow", self.__language_box[self.__language][1]))
        self.NoArrowsText.setText(_translate("MainWindow", self.__language_box[self.__language][2]))
        self.NoPointsText.setText(_translate("MainWindow", self.__language_box[self.__language][27]))
        self.PreparationTimeText.setText(_translate("MainWindow", self.__language_box[self.__language][28]))
        self.ShootTimeText.setText(_translate("MainWindow", self.__language_box[self.__language][36]))
        self.GameTypeText.setText(_translate("MainWindow", self.__language_box[self.__language][37]))
        for i, language in enumerate(self.__language_box.columns):
            self.LanguageComboBox.setItemText(i, _translate("MainWindow", language))
            self.LanguageComboBox.setItemText(i, _translate("MainWindow", language))
        self.TargetComboBox.setItemText(0, _translate("MainWindow", "REGULAR_1_10"))
        self.TargetComboBox.setItemText(1, _translate("MainWindow", "REGULAR_5_10"))
        self.TargetComboBox.setItemText(2, _translate("MainWindow", "REGULAR_6_10"))
        self.GameTypeComboBox.setItemText(0, _translate("MainWindow", self.__language_box[self.__language][38]))
        self.GameTypeComboBox.setItemText(1, _translate("MainWindow", self.__language_box[self.__language][39]))
        self.NoPlayersLineEdit.setText(_translate("MainWindow", "2"))
        self.NoSetsLineEdit.setText(_translate("MainWindow", "5"))
        self.PreparationTimeLineEdit.setText(_translate("MainWindow", "5"))
        self.ShootTimeLineEdit.setText(_translate("MainWindow", "20"))
        self.NoArrowsLineEdit.setText(_translate("MainWindow", "3"))
        self.NoPointsLineEdit.setText(_translate("MainWindow", "6"))
        self.NextPushButton.setText(_translate("MainWindow", self.__language_box[self.__language][3]))
        self.BackPushButton.setText(_translate("MainWindow", self.__language_box[self.__language][6]))
        self.PickTargetText.setText(_translate("MainWindow", self.__language_box[self.__language][7]))
        self.TargetImage.setText(_translate("MainWindow", ""))
        self.IPAddrText.setText(_translate("MainWindow", self.__language_box[self.__language][11]))
        self.DroidCamPortText.setText(_translate("MainWindow", self.__language_box[self.__language][12]))
        self.ConnectPushButton.setText(_translate("MainWindow", self.__language_box[self.__language][10]))
        self.DisconnectPushButton.setText(_translate("MainWindow", self.__language_box[self.__language][22]))
        self.DetectPushButton.setText(_translate("MainWindow", self.__language_box[self.__language][18]))
        self.ManualMarkPushButton.setText(_translate("MainWindow", self.__language_box[self.__language][19]))
        self.ResetTargetPushButton.setText(_translate("MainWindow", self.__language_box[self.__language][26]))
        self.NextSetPushButton.setText(_translate("MainWindow", self.__language_box[self.__language][29]))
        self.TimeDisplaySquare.setText(_translate("MainWindow",""))
        self.TimeDisplayText.setText(_translate("MainWindow",""))
        self.WinnerDisplayText.setText(_translate("MainWindow", ""))

    def __proceedGame(self):
        prevState = self.__gameState
        self.__gameState = self.__game.proceedGame()
        if prevState != self.__gameState:
            if prevState == GameState.BREAK and self.__gameState == GameState.READY_GO:
                self.__proceed = True
                self.NextSetPushButton.hide()
                self.TimeDisplaySquare.setPixmap(QtGui.QPixmap('images/yellow_background.jpg'))
            if prevState == GameState.READY_GO and self.__gameState == GameState.ROUND:
                self.TimeDisplaySquare.setPixmap(QtGui.QPixmap('images/green_background.jpg'))
            if prevState == GameState.ROUND and self.__gameState == GameState.READY_GO:
                s,p,a = self.__game.getTokens()
                self.ScoreTable.setItem(s, (p-1)*(self.__noArrows+1)+a, QTableWidgetItem(str(self.__game.getHitTable()[s-1,p-1,a-1])))
                self.ScoreTable.setItem(s, p*(self.__noArrows+1), QTableWidgetItem(str(self.__game.getSumTable()[s-1,p-1])))
                self.TimeDisplaySquare.setPixmap(QtGui.QPixmap('images/yellow_background.jpg'))
            if prevState == GameState.ROUND and self.__gameState == GameState.BREAK:
                s,p,a = self.__game.getTokens()
                self.ScoreTable.setItem(s, (p-1)*(self.__noArrows+1)+a, QTableWidgetItem(str(self.__game.getHitTable()[s-1,p-1,a-1])))
                self.ScoreTable.setItem(s, p*(self.__noArrows+1), QTableWidgetItem(str(self.__game.getSumTable()[s-1,p-1])))
                self.__proceed = False
                self.TimeDisplaySquare.setPixmap(QtGui.QPixmap('images/red_background.jpg'))
                score = self.__game.getScoreTable()
                for i, sc in enumerate(score):
                    self.ScoreTable.setItem(0,(i+1)*(self.__noArrows+1),QTableWidgetItem(str(sc)))
                self.NextSetPushButton.show()
            if prevState == GameState.BREAK and self.__gameState == GameState.GAME_OVER:
                self.__proceed = False
                self.TimeDisplaySquare.setPixmap(QtGui.QPixmap('images/red_background.jpg'))
                self.TimeDisplayText.setText('-')

                winners = np.where(self.__game.getScoreTable() == np.max(self.__game.getScoreTable()))
                if len(winners) == 1:
                    self.WinnerDisplayText.setText(self.__language_box[self.__language][40]+str(winners[0]+1)+" "+self.__language_box[self.__language][41])
                    self.WinnerDisplayText.show()
                else:
                    txt = self.__language_box[self.__language][42] + " " + self.__language_box[self.__language][40] + str(winners[0]+1)
                    
                    for win in winners[1:]:
                        txt += " " + self.__language_box[self.__language][43] + " " + self.__language_box[self.__language][40] + str(win+1)

                self.WinnerDisplayText.show()
        if self.__gameState == GameState.ROUND:
            dist = self.__model.getHit(self.__prevPrevFrame,\
                                       self.__prevFrame,\
                                        self.__actFrame)
            if dist is not None:
                self.__game.passHit(dist)

                print("returned: "+str(dist))
                
        return self.__gameState

    def __changeLanguage(self):
        if self.__WindowIndex == 0:
            self.__language = self.LanguageComboBox.currentText()
            self.retranslateUi(self)

    def __changeTarget(self):
        self.TargetImage.setPixmap(QtGui.QPixmap(self.__TargetPhotosPaths[self.TargetComboBox.currentIndex()]))
        if int(self.TargetComboBox.currentIndex()) == 0:
            self.__targetType = TargetType.REGULAR_1_10
        if int(self.TargetComboBox.currentIndex()) == 1:
            self.__targetType = TargetType.REGULAR_5_10
        if int(self.TargetComboBox.currentIndex()) == 2:
            self.__targetType = TargetType.REGULAR_6_10

    def __changeGameType(self):
        if int(self.GameTypeComboBox.currentIndex()) == 0:
            self.__gameType = GameType.TO_SPECIFIED_AMOUNT_OF_SETS
        if int(self.GameTypeComboBox.currentIndex()) == 1:
            self.__gameType = GameType.TO_SPECIFIED_AMOUNT_OF_POINTS

    def __detectTarget(self):
        self.ManualMarkPushButton.hide()
        self.__detect = True

    def __disconnectCamera(self):
        self.__disconnect = True

    def __markPoints(self, event, x, y, flags, param):        
        if event == cv2.EVENT_LBUTTONDOWN and len(self.__markedPoints) < 5:
            self.__markedPoints.append((x,y))
        if event == cv2.EVENT_RBUTTONDOWN and len(self.__markedPoints) > 0:
            self.__markedPoints.pop()
        if len(self.__markedPoints) == 5:
            self.__ellipse = self.__model.createEllipse(self.__markedPoints)
            self.__model.prepareTransformation(self.__actFrame)
        else:
            self.__ellipse = None

    def __markTarget(self):
        self.DetectPushButton.hide()
        self.ManualMarkPushButton.hide()
        self.ResetTargetPushButton.show()
        self.__markedPoints = []
        self.__manualMark = True

        frame, status = captureVideo(self.__Video)

        if status == ConnectionStatus.ERROR:
            return
        
        markPointsInfo = QMessageBox(self)
        markPointsInfo.setIcon(QMessageBox.Information)
        markPointsInfo.setWindowTitle("")
        markPointsInfo.setText(self.__language_box[self.__language][23])
        markPointsInfo.setStandardButtons(QMessageBox.Ok)
        markPointsInfo.exec_()
        
        cv2.imshow("CameraView", frame)
        cv2.setMouseCallback("CameraView", self.__markPoints)

    def __resetTarget(self):
        self.__ellipse = None
        self.__detect = False
        self.__manualMark = False
        self.__markedPoints = []

        self.__model.resetEllipse()

        self.DetectPushButton.show()
        self.ManualMarkPushButton.show()
        self.ResetTargetPushButton.hide()
        

    def __connectToCamera(self):
        if self.__game is None:
            self.__game = Game(self.__noPlayers, self.__gameType, self.__noSets, self.__noPoints, self.__noArrows, self.__targetType,\
                           self.__preparationTime, self.__shootTime)
        if self.__model is None:
            self.__model = ArcheryTargetModel(self.__targetType)

        # IP and port validation
        if any(elem =='' for elem in[self.IPLineEdit1.text(), self.IPLineEdit2.text(), self.IPLineEdit3.text(), 
                self.IPLineEdit4.text(), self.DroidCamPortLineEdit.text()]):
            
            warning_message = QMessageBox(self)
            warning_message.setIcon(QMessageBox.Warning)
            warning_message.setWindowTitle(self.__language_box[self.__language][4])
            warning_message.setText(self.__language_box[self.__language][13])
            warning_message.setStandardButtons(QMessageBox.Ok)
            warning_message.exec_()
        else:
            # video capturing
            url = "http://"+self.IPLineEdit1.text()+"."+self.IPLineEdit2.text()+"."+self.IPLineEdit3.text()+\
                "."+self.IPLineEdit4.text()+":"+self.DroidCamPortLineEdit.text()+"/video"
            
            self.__Video  = cv2.VideoCapture(url)
            frame, status = captureVideo(self.__Video)

            self.__actFrame = frame.copy()
            
            if status == ConnectionStatus.ERROR:
                # error service
                error_message = QMessageBox(self)
                error_message.setIcon(QMessageBox.Critical)
                error_message.setWindowTitle(self.__language_box[self.__language][14])
                error_message.setText(self.__language_box[self.__language][15]+'\n'+self.__language_box[self.__language][16]+\
                                    '\n'+self.__language_box[self.__language][17])
                error_message.setStandardButtons(QMessageBox.Ok)
                error_message.exec_()    
            else:
                self.__connected = True   
                self.ConnectPushButton.hide()
                self.DisconnectPushButton.show() 
                self.DetectPushButton.show()
                self.ManualMarkPushButton.show()

            i = 0

            while status == ConnectionStatus.OK and self.__disconnect == False:
                i += 1
                if i % 100 == 0:
                    if self.__proceed:
                        self.__proceedGame()

                        self.TimeDisplayText.setText(str(int(self.__game.getTimer())))

                    imshow = frame.copy()

                    # if target is detected/marked draw it
                    if self.__ellipse is not None:
                        imshow = self.__model.drawEllipse(imshow)
                    
                    # when target is pointed manually, draw marked points
                    if self.__manualMark:
                        imshow = self.__model.drawPoints(imshow, self.__markedPoints)
                        if len(self.__markedPoints) == 5 and self.__ellipse is not None:
                            self.__model.prepareTransformation(frame.copy())

                    # constant image display
                    if self.__WindowIndex == 3 and self.__ellipse is not None and imshow is not None:
                        imshow = self.__model.getTransformedImage(frame)
                        if self.__actFrame is None:
                            self.__actFrame = imshow
                            self.__prevFrame = imshow
                            self.__prevPrevFrame = imshow
                        self.__prevPrevFrame = self.__prevFrame
                        self.__prevFrame = self.__actFrame
                        self.__actFrame = imshow
                        cv2.imshow("CameraView", imshow)
                    if self.__WindowIndex == 2 and imshow is not None:
                        cv2.imshow("CameraView", imshow)

                    """if self.__proceed:
                        if self.__actFrame is None:
                            self.__actFrame = imshow
                            self.__prevFrame = imshow
                            self.__prevPrevFrame = imshow
                        self.__prevPrevFrame = self.__prevFrame
                        self.__prevFrame = self.__actFrame
                        self.__actFrame = imshow
                        if self.__actFrame.shape != self.__prevFrame.shape:
                            self.__prevFrame = imshow
                            self.__prevPrevFrame = imshow
                        if self.__actFrame.shape != self.__prevPrevFrame.shape:
                            self.__prevPrevFrame = self.__prevFrame
                    else:
                        self.__actFrame = frame.copy()"""
                    i=0

                    frame, status = captureVideo(self.__Video)

                    if (cv2.waitKey(1) & 0xFF == ord('q')) or self.__disconnect:
                        break
                    
                    # when order to detect target
                    if self.__detect:
                        #self.__ellipse = self.__model.detectTarget(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
                        fRGB = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                        self.__ellipse = self.__model.detectTarget(fRGB)
                        if self.__ellipse is None:
                            # if target was not found
                            errorEllNotFound = QMessageBox(self)
                            errorEllNotFound.setIcon(QMessageBox.Critical)
                            errorEllNotFound.setWindowTitle(self.__language_box[self.__language][14])
                            errorEllNotFound.setText(self.__language_box[self.__language][20])
                            errorEllNotFound.setStandardButtons(QMessageBox.Ok)
                            errorEllNotFound.exec_()
                            
                            self.ManualMarkPushButton.show()
                        else:
                            self.DetectPushButton.hide()
                            self.ResetTargetPushButton.show()
                        self.__detect = False

            self.DisconnectPushButton.hide()
            self.DetectPushButton.hide()
            self.ManualMarkPushButton.hide()
            self.ConnectPushButton.show()

            self.__disconnect = False

            self.__Video.release()
            cv2.destroyAllWindows()

    def __createScoreTable(self):
        self.ScoreTable = QtWidgets.QTableWidget()

        class BoldBorderDelegate(QtWidgets.QStyledItemDelegate):
            def __init__(self, noPlayers, noSets, noArrows, parent=None):
                super().__init__(parent)
                self.__noPlayers = noPlayers
                self.__noSets = noSets
                self.__noArrows = noArrows

            def paint(self, painter, option, index):
                super().paint(painter, option, index)

                painter.save()
                pen = QtGui.QPen(QtGui.QColor("black"))

                if not(index.row() == 0 or index.row() == self.__noSets):
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
                else:
                    pen.setWidth(4)
                    painter.setPen(pen)
                    painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
                    if index.row() == 0:
                        painter.drawLine(option.rect.topLeft(), option.rect.topRight())
                if index.column() % (self.__noArrows+1) == 0:
                    pen.setWidth(4)
                    painter.setPen(pen)
                    painter.drawLine(option.rect.topRight(), option.rect.bottomRight())
                    if index.column() != 0:
                        pen.setWidth(2)
                        painter.setPen(pen)
                        painter.drawLine(option.rect.topLeft(), option.rect.bottomLeft())
                    else:
                        painter.drawLine(option.rect.topLeft(), option.rect.bottomLeft())

                    
                painter.restore()

        self.ScoreTable.setColumnCount(1+(self.__noArrows+1)*self.__noPlayers)
        self.ScoreTable.setRowCount(2+self.__noSets)
        self.ScoreTable.setItem(0,0,QtWidgets.QTableWidgetItem('Sets\Players+Score'))
        for i in range(self.__noPlayers):
            self.ScoreTable.setItem(0,((self.__noArrows+1)*i)+1,QtWidgets.QTableWidgetItem('Player '+str(i+1)))
            self.ScoreTable.setSpan(0,i*(self.__noArrows+1)+1, 1, self.__noArrows)
        for i in range(self.__noSets):
            self.ScoreTable.setItem(i+1,0,QtWidgets.QTableWidgetItem('Set '+str(i+1)))

        self.delegate = BoldBorderDelegate(self.__noPlayers, self.__noSets, self.__noArrows, self.ScoreTable)
        self.ScoreTable.setItemDelegate(self.delegate)                    

    def __NextWindow(self):
        if self.__WindowIndex == 2:
            if self.__connected and self.__ellipse is not None:
                self.IPAddrText.hide()
                self.DroidCamPortText.hide()
                self.IPLineEdit1.hide()
                self.IPLineEdit2.hide()
                self.IPLineEdit3.hide()
                self.IPLineEdit4.hide()
                self.DroidCamPortLineEdit.hide()
                self.ConnectPushButton.hide()
                self.DetectPushButton.hide()
                self.ManualMarkPushButton.hide()
                self.ResetTargetPushButton.hide()
                self.BackPushButton.hide()

                self.__detect = False
                self.__manualMark = False
                self.__markedPoints = []

                if self.ScoreTable is None:
                    self.__createScoreTable()
                self.ScoreTable.show()
                self.TimeDisplaySquare.show()
                if self.TimeDisplayText.text() == "":
                    self.TimeDisplayText.setText(str(self.__preparationTime))
                self.TimeDisplayText.show()
                self.NextSetPushButton.show()
                
                self.__WindowIndex = 3
            else:
                errorEllNotFound = QMessageBox(self)
                errorEllNotFound.setIcon(QMessageBox.Critical)
                errorEllNotFound.setWindowTitle(self.__language_box[self.__language][4])
                errorEllNotFound.setText(self.__language_box[self.__language][24]+'\n'+self.__language_box[self.__language][25])
                errorEllNotFound.setStandardButtons(QMessageBox.Ok)
                errorEllNotFound.exec_()             

        if self.__WindowIndex == 1:
            self.PickTargetText.hide()
            self.TargetComboBox.hide()
            self.TargetImage.hide()

            self.IPAddrText.show()
            self.DroidCamPortText.show()
            self.IPLineEdit1.show()
            self.IPLineEdit2.show()
            self.IPLineEdit3.show()
            self.IPLineEdit4.show()
            self.DroidCamPortLineEdit.show()
            self.ConnectPushButton.show()

            self.__WindowIndex = 2
        if self.__WindowIndex == 0:
            if self.NoPlayersLineEdit.text() == '' or self.NoPointsLineEdit.text() == '' or self.NoSetsLineEdit.text() == ''\
                or self.NoArrowsLineEdit.text() == '' or self.PreparationTimeLineEdit.text() == '' or self.ShootTimeLineEdit.text() == '':
                warning_message = QMessageBox(self)
                warning_message.setIcon(QMessageBox.Warning)
                warning_message.setWindowTitle(self.__language_box[self.__language][4])
                warning_message.setText(self.__language_box[self.__language][5]+self.__language_box[self.__language][28])
                warning_message.setStandardButtons(QMessageBox.Ok)
                warning_message.exec_()
            else:
                self.NoPlayersLineEdit.hide()
                self.NoSetsLineEdit.hide()  
                self.NoArrowsLineEdit.hide() 
                self.NoPlayersText.hide()
                self.NoSetsText.hide()
                self.NoArrowsText.hide()
                self.PreparationTimeLineEdit.hide()
                self.PreparationTimeText.hide()
                self.ShootTimeLineEdit.hide()
                self.ShootTimeText.hide()
                self.GameTypeComboBox.hide()
                self.GameTypeText.hide()
                self.NoPointsLineEdit.hide()
                self.NoPointsText.hide()

                self.BackPushButton.show()
                self.PickTargetText.show()
                self.TargetComboBox.show()
                self.TargetImage.show()

                self.__noPlayers = int(self.NoPlayersLineEdit.text())
                self.__noSets = int(self.NoSetsLineEdit.text())
                self.__noPoints = int(self.NoPointsLineEdit.text())
                self.__noArrows = int(self.NoArrowsLineEdit.text())
                self.__preparationTime = int(self.PreparationTimeLineEdit.text())
                self.__shootTime = int(self.ShootTimeLineEdit.text())

                self.__WindowIndex = 1

    def __PrevWindow(self):
        if self.__WindowIndex == 1:
            self.BackPushButton.hide()
            self.PickTargetText.hide()
            self.TargetComboBox.hide()
            self.TargetImage.hide()

            self.NoPlayersLineEdit.show()
            self.NoSetsLineEdit.show()  
            self.NoArrowsLineEdit.show() 
            self.NoPlayersText.show()
            self.NoSetsText.show()
            self.NoArrowsText.show()
            self.PreparationTimeLineEdit.show()
            self.PreparationTimeText.show()
            self.ShootTimeLineEdit.show()
            self.ShootTimeText.show()
            self.GameTypeComboBox.show()
            self.GameTypeText.show()
            self.NoPointsLineEdit.show()
            self.NoPointsText.show()
            
            self.__WindowIndex = 0
        
        if self.__WindowIndex == 2:
            self.IPAddrText.hide()
            self.DroidCamPortText.hide()
            self.IPLineEdit1.hide()
            self.IPLineEdit2.hide()
            self.IPLineEdit3.hide()
            self.IPLineEdit4.hide()
            self.DroidCamPortLineEdit.hide()
            self.ConnectPushButton.hide()
            self.DisconnectPushButton.hide()
            self.DetectPushButton.hide()
            self.ManualMarkPushButton.hide()
            self.ResetTargetPushButton.hide()

            self.BackPushButton.show()
            self.PickTargetText.show()
            self.TargetComboBox.show()
            self.TargetImage.show()

            self.__WindowIndex = 1

        if self.__WindowIndex == 3:
            self.ScoreTable.hide()
            self.TimeDisplaySquare.hide()
            self.TimeDisplayText.hide()
            self.NextSetPushButton.hide()

            self.IPAddrText.show()
            self.DroidCamPortText.show()
            self.IPLineEdit1.show()
            self.IPLineEdit2.show()
            self.IPLineEdit3.show()
            self.IPLineEdit4.show()
            self.DroidCamPortLineEdit.show()
            self.ConnectPushButton.show()
            self.DetectPushButton.show()
            self.ManualMarkPushButton.show()
            self.ResetTargetPushButton.show()

            self.__ellipse = self.__backupEllipse
            self.__backupEllipse = None

            self.__WindowIndex = 2
            

        
        

