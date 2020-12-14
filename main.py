# -*- coding: utf-8 -*-

# from matplotlib.pyplot import plot, 
from pandas import read_excel, read_csv
from numpy import random, zeros, where, genfromtxt, arange, squeeze, expand_dims
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
from src.model import CoffeeModel
from lib.model import SVM, RF, NIRResNet152

flavor_colors = ['rgb(221, 9, 103)'] * 2 + ['rgb(221, 26, 29)'] * 5 + ['rgb(237, 181, 3)'] * 3 + ['rgb(21, 123, 43)'] * 2 + \
    ['rgb(6, 163, 183)'] * 2 + ['rgb(203, 72, 41)'] * 2 + ['rgb(175, 31, 59)'] + ['rgb(169, 123, 98)'] * 2 + ['rgb(233, 87, 39)'] * 2

stylesheet = """
    MainWindow {
        background-image: url("./src/icons/ssl.png"); 
        background-repeat: no-repeat; 
        background-position: center;
    }

    AboutWindow {
        background-image: url("./src/icons/ssl.png"); 
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
            \nUpdated 1091215 ")
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
        self.setWindowIcon(QIcon('./lib/icons/ssl_2.png')) 
        self.setWindowTitle('Nice Coffee App')
        # self.cwd = getcwd()

        self.agtron_model = None
        self.flavor_model = SVM().load('./SVM.svm')

        self.waves = {}
        self.flavor_labels = [
            self.label_floral, self.label_tealike, self.label_tropical, self.label_stone, self.label_citrus, self.label_berry, self.label_other, \
            self.label_sour, self.label_alcohol, self.label_fermented, self.label_fresh, self.label_dry, self.label_papery, self.label_chemical, \
            self.label_burnt, self.label_cereal, self.label_spices, self.label_nutty, self.label_cocoa, self.label_sweet, self.label_butter
            ]

        self.comboBox_flavor_model.addItems(['Support vector machine (SVM)', 'Random forest (RF)', 'ResNet152focal (DCNN)'])
        self.comboBox_flavor_model.currentTextChanged.connect(self.select_flavor_model)
        self.comboBox_agtron_model.addItems(['Narrow (700-900, 1160-1260)', 'Wide (700-2500)', 'CARS (selected wavenumbers)', 'ResNet18 (DCNN)'])
        self.comboBox_agtron_model.currentTextChanged.connect(self.select_agtron_model)
        self.comboBox_spectrum.setPlaceholderText('Open or choose a spectrum')
        self.comboBox_spectrum.currentTextChanged.connect(self.select_spectrum)
        
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
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,  
            "Open a spectrum file",  
            "",  
            "Spectrum files (*.csv *.xlsx)"
            )
        if file_paths == []:
            return

        # if file_path.endswith('.txt'):
        #     df = read_csv(file_path, header=None)
        #     self.file_name = file_path.split('/')[-1]
        #     df = df.to_numpy()
        #     print(df)
        #     self.wave = squeeze(df)
        #     self.plot_wave()

        for file in file_paths:

            if file.endswith('.csv'):
                df = read_csv(file, header=None)
                self.file_name = file.split('/')[-1]
                df = df.to_numpy()
                self.wave = squeeze(df)

            elif file.endswith('.xlsx'):
                df = read_excel(file, header=None)
                self.file_name = file.split('/')[-1]
                df = df.to_numpy()
                self.wave = squeeze(df)
            
            # if self.waves == {}:
            #     self.plot_wave()

            if self.file_name not in self.waves:
                self.comboBox_spectrum.addItem(self.file_name)
                self.waves[self.file_name] = self.wave

    def save_file(self, e):
        '''
        Save coffee infomation
        '''
        file_path, _ = QFileDialog.getSaveFileName(
            self,  
            "Save as",  
            "",
            "Text files (*.csv *.xlsx)"
            )  

        if file_path == "":
            return

        else:
            print(file_path)

    def open_about_window(self):
        self.about_window = AboutWindow(self)
        self.about_window.show()

    def predict(self):
        '''
        Get the Agtron and flavor from predicting models and pass into the label widgets
        '''
        wave = expand_dims(self.wave, axis=0).copy()

        '''
        Predict Agtron number
        '''
        print(wave[-1][-1])
        pred_agtron = self.agtron_model.predict(wave)
        self.label_agtron.setText(QCoreApplication.translate("MainWindow", f"{pred_agtron:.1f}"))
        self.label_agtron.setStatusTip(QCoreApplication.translate("MainWindow", f"Agtron number: {pred_agtron:.1f}"))
        
        '''
        Predict flavor
        '''
        pred_flavor = self.flavor_model.predict(wave)[0]
        for f in range(len(pred_flavor)):
            if pred_flavor[f] == 1:
                self.flavor_labels[f].setStyleSheet(f"background-color: {flavor_colors[f]};")
            elif pred_flavor[f] == 0:
                self.flavor_labels[f].setStyleSheet("background-color: rgb(255, 255, 255);")

    def select_flavor_model(self, model_name):
        if model_name.startswith('Support'):
            self.flavor_model = SVM().load('./SVM.svm')
        if model_name.startswith('Random'):
            self.flavor_model = RF().load('./RF.rf')
        if model_name.startswith('ResNet'):
            pass

    def select_agtron_model(self, model_name):
        if model_name.startswith('Narrow'):
            pass
        if model_name.startswith('Wide'):
            pass
        if model_name.startswith('CARS'):
            pass
        if model_name.startswith('ResNet'):
            model = CoffeeModel()
            model.load_networks('./ResNet18.pth')
            self.agtron_model = model

    def select_spectrum(self, spectrum_name):
        if self.waves == {}:
            return
        self.wave = self.waves[spectrum_name]
        self.file_name = spectrum_name
        self.plot_wave()
    
    def plot_wave(self):
        self.widget_preview.axes.cla()
        self.widget_preview.axes.plot(arange(700, 2500, 2), self.wave)
        self.widget_preview.axes.set_xticks(list(range(700, 2501, 200)))
        self.widget_preview.axes.set_title(self.file_name)
        # self.widget_preview.axes.set_axis_off()
        self.widget_preview.draw()

    

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