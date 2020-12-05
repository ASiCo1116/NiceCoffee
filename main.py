# -*- coding: utf-8 -*-

from sys import argv
from os import getcwd
from PyQt5.QtGui import (QColor)
from PyQt5.QtCore import (QSize)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QColorDialog, QToolBar, QAction)

from mainwindow import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.cwd = getcwd()

        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)

        toolbar.setIconSize(QSize(40, 40))

        self.action_open_spectrum.setStatusTip("Open a spectrum file")
        self.action_open_spectrum.triggered.connect(self.onMyToolBarButtonClick)
        toolbar.addAction(self.action_open_spectrum)

        self.action_save_as.setStatusTip("Save coffee infomation")
        self.action_save_as.triggered.connect(self.onMyToolBarButtonClick)
        toolbar.addAction(self.action_save_as)

        self.action_about.setStatusTip("About this app")
        self.action_about.triggered.connect(self.onMyToolBarButtonClick)
        toolbar.addAction(self.action_about)

    def onMyToolBarButtonClick(self, s):
        print("click", s)

app = QApplication(argv)

window = MainWindow()
window.show()
app.exec()