# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, dpi=75):
        self.fig = Figure(figsize=(1., 1.), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_axis_off()
        super().__init__(self.fig)
