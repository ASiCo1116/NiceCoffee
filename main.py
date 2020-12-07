# -*- coding: utf-8 -*-


from numpy import random, zeros, where
from sys import argv
from os import getcwd
from PyQt5.QtGui import (
    QColor,
    QPixmap,
    QIcon,
    QPalette
    )
from PyQt5.QtCore import (
    QSize,
    Qt,
    QCoreApplication
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
    QWidget,
    QStyle
    )
from mainwindow import Ui_MainWindow

flavor_colors = ['rgb(221, 9, 103)'] * 2 + ['rgb(221, 26, 29)'] * 5 + ['rgb(237, 181, 3)'] * 3 + ['rgb(21, 123, 43)'] * 2 + \
    ['rgb(6, 163, 183)'] * 2 + ['rgb(203, 72, 41)'] * 2 + ['rgb(175, 31, 59)'] + ['rgb(169, 123, 98)'] * 2 + ['rgb(233, 87, 39)'] * 2

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

def set_label_color(widget, color):
    widget.setStyleSheet(f"background-color: color;")

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
        '''
        Initialize the interface
        '''
        self.setupUi(self)
        self.setWindowIcon(QIcon('./icons/ssl_2.png')) 
        self.setWindowTitle('Nice Coffee App')
        # self.cwd = getcwd()

        self.flavor_labels = [
            self.label_floral, self.label_tealike, self.label_tropical, self.label_stone, self.label_citrus, self.label_berry, self.label_other, \
            self.label_sour, self.label_alcohol, self.label_fermented, self.label_fresh, self.label_dry, self.label_papery, self.label_chemical, \
            self.label_burnt, self.label_cereal, self.label_spices, self.label_nutty, self.label_cocoa, self.label_sweet, self.label_butter
            ]

        '''
        Set the predict button
        '''
        style = self.pushButton_predict.style() # Get the QStyle object from the widget.
        icon = style.standardIcon(QStyle.SP_MediaPlay)
        self.pushButton_predict.setIcon(icon)
        self.pushButton_predict.clicked.connect(self.predict)

        '''
        Initialize the toolbar
        '''

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

    def predict(self, flavor_model = None , agtron_model = None) -> None:
        '''Get the Agtron and flavor from predicting models and pass into the label widgets\n
        Parameters:flavor_model, agtron_model
        '''
        '''
        Testing Agtron number
        '''
        pred_agtron = round(random.random() * (106.0 - 60.0) + 60.0, 1)
        self.label_agtron.setText(QCoreApplication.translate("MainWindow", f"{pred_agtron:}"))
        self.label_agtron.setStatusTip(QCoreApplication.translate("MainWindow", f"Agtron number: {pred_agtron:}"))

        '''
        Testing flavor
        '''
        pred_flavor = random.randint(2, size=21)
        for f in range(pred_flavor.shape[0]):
            if pred_flavor[f] == 1:
                self.flavor_labels[f].setStyleSheet(f"background-color: {flavor_colors[f]};")
            elif pred_flavor[f] == 0:
                self.flavor_labels[f].setStyleSheet("background-color: rgb(255, 255, 255);")

def main():

    # darkPalette = QPalette()
    # darkPalette.setColor(QPalette.Window, QColor(53, 53, 53))
    # darkPalette.setColor(QPalette.WindowText, Qt.white)
    # darkPalette.setColor(QPalette.Disabled, QPalette.WindowText, QColor
    # (127, 127, 127))
    # darkPalette.setColor(QPalette.Base, QColor(42, 42, 42))
    # darkPalette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
    # darkPalette.setColor(QPalette.ToolTipBase, Qt.white)
    # darkPalette.setColor(QPalette.ToolTipText, Qt.white)
    # darkPalette.setColor(QPalette.Text, Qt.white)
    # darkPalette.setColor(QPalette.Disabled, QPalette.Text, QColor(127,
    # 127, 127))
    # darkPalette.setColor(QPalette.Dark, QColor(35, 35, 35))
    # darkPalette.setColor(QPalette.Shadow, QColor(20, 20, 20))
    # darkPalette.setColor(QPalette.Button, QColor(53, 53, 53))
    # darkPalette.setColor(QPalette.ButtonText, Qt.white)
    # darkPalette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor
    # (127, 127, 127))
    # darkPalette.setColor(QPalette.BrightText, Qt.red)
    # darkPalette.setColor(QPalette.Link, QColor(42, 130, 218))
    # darkPalette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    # darkPalette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80,
    # 80, 80))
    # darkPalette.setColor(QPalette.HighlightedText, Qt.white)
    # darkPalette.setColor(QPalette.Disabled, QPalette.HighlightedText,
    # QColor(127, 127, 127))
    
    app = QApplication(argv)
    app.setStyleSheet(stylesheet)
    app.setStyle('Fusion')
    # app.setPalette(darkPalette)

    window = MainWindow()
    window.show()

    # qtmodern.styles.dark(app)
    # mw = qtmodern.windows.ModernWindow(window)
    # mw.show()
    app.exec()

if __name__ == "__main__":
    main()