# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(600, 600)
        sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(600, 600))
        self.gridLayout = QGridLayout(MainWindow)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout_main = QVBoxLayout()
        self.verticalLayout_main.setObjectName(u"verticalLayout_main")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.regenerate_button = QPushButton(MainWindow)
        self.regenerate_button.setObjectName(u"regenerate_button")

        self.horizontalLayout.addWidget(self.regenerate_button)

        self.label_2 = QLabel(MainWindow)
        self.label_2.setObjectName(u"label_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy1)
        self.label_2.setMinimumSize(QSize(80, 0))

        self.horizontalLayout.addWidget(self.label_2)

        self.dataID = QLineEdit(MainWindow)
        self.dataID.setObjectName(u"dataID")

        self.horizontalLayout.addWidget(self.dataID)

        self.start_saving_button = QPushButton(MainWindow)
        self.start_saving_button.setObjectName(u"start_saving_button")
        self.start_saving_button.setCheckable(True)

        self.horizontalLayout.addWidget(self.start_saving_button)

        self.quit_button = QPushButton(MainWindow)
        self.quit_button.setObjectName(u"quit_button")

        self.horizontalLayout.addWidget(self.quit_button)


        self.verticalLayout_main.addLayout(self.horizontalLayout)

        self.label = QLabel(MainWindow)
        self.label.setObjectName(u"label")
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QSize(500, 500))
        self.label.setMaximumSize(QSize(10240, 7680))
        self.label.setSizeIncrement(QSize(1, 1))
        self.label.setFrameShape(QFrame.Box)
        self.label.setScaledContents(True)

        self.verticalLayout_main.addWidget(self.label)


        self.gridLayout.addLayout(self.verticalLayout_main, 0, 0, 1, 1)


        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Social Navigation Dataset Generator", None))
        self.regenerate_button.setText(QCoreApplication.translate("MainWindow", u"regenerate", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"data id", None))
        self.dataID.setText(QCoreApplication.translate("MainWindow", u"A", None))
        self.start_saving_button.setText(QCoreApplication.translate("MainWindow", u"start saving", None))
        self.quit_button.setText(QCoreApplication.translate("MainWindow", u"quit", None))
        self.label.setText("")
    # retranslateUi

