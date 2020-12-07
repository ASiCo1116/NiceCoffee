# -*- coding: utf-8 -*-

from sys import argv
from os import getcwd
from PyQt5.QtGui import (
    QColor,
    QPixmap
    )
from PyQt5.QtCore import (
    QSize,
    Qt
    )
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QColorDialog,
    QToolBar,
    QAction,
    QDialog,
    QLabel,
    QVBoxLayout,
    QWidget
    )
from mainwindow import Ui_MainWindow

stylesheet ="""
    MainWindow {
        background-image: url("./icons/ssl.png"); 
        background-repeat: no-repeat; 
        background-position: center;
    }

    AboutWindow {
        background-image: url("./icons/ssl.png"); 
        background-repeat: no-repeat; 
        background-position: center;
    }
"""

class AboutWindow(QMainWindow):
    def __init__(self, parent=None):
        super(AboutWindow, self).__init__(parent)
        self.setWindowTitle("About Nice Coffee App")
        self.setFixedSize(QSize(600, 450))

        copyright = QLabel("Copyright © 2020 All Rights Reserved \
            \nAuthors: Meng-Chien Hsueh, Yu-Tang Chang & Shih-Fang Chen\
            \nSensing and Spectroscopy Lab (SSL) \
            \nDepartment of BioMechtronics (BiME)\
            \nNational Taiwan University (NTU) \
            \nUpdated 1091205 ")
        font = copyright.font()
        
        copyright.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        layout = QVBoxLayout()
        layout.addWidget(copyright)

        widget = QWidget()
        widget.setLayout(layout)
        
        self.setCentralWidget(widget)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowTitle('Nice Coffee App')

        # self.cwd = getcwd()

        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)
        toolbar.setIconSize(QSize(40, 40))

        self.action_open_spectrum.setStatusTip("Open a spectrum file")
        self.action_open_spectrum.triggered.connect(self.open_file)
        toolbar.addAction(self.action_open_spectrum)

        self.action_save_as.setStatusTip("Save coffee infomation")
        self.action_save_as.triggered.connect(self.save_file)
        toolbar.addAction(self.action_save_as)

        self.action_about.setStatusTip("About this app")
        self.action_about.triggered.connect(self.open_about_window)
        toolbar.addAction(self.action_about)

    def open_file(self, e):
        '''
        Open a spectrum file
        '''
        file_path, _ = QFileDialog.getOpenFileName(
            self,  
            "Open a spectrum file",  
            "",  
            "Text files (*.txt *.csv *.xlsx)"
            ) 

        if file_path == "":
            return

        else:
            print(file_path)
    
    def save_file(self, e):
        '''
        Save coffee infomation
        '''
        file_path, _ = QFileDialog.getSaveFileName(
            self,  
            "Save as",  
            "",
            "Text files (*.txt *.csv *.xlsx)"
            )  

        if file_path == "":
            return

        else:
            print(file_path)

    def open_about_window(self):
        self.about_window = AboutWindow(self)
        self.about_window.show()

def main():
    app = QApplication(argv)
    app.setStyleSheet(stylesheet)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()