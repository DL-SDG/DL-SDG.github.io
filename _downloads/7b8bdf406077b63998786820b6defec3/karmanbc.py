import sys
import os.path
from PyQt6 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
import matplotlib as mpl
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 552)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.wallsButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.wallsButton.setGeometry(QtCore.QRect(20, 230, 141, 51))
        self.wallsButton.setObjectName("wallsButton")
        self.clearButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.clearButton.setGeometry(QtCore.QRect(640, 470, 141, 51))
        self.clearButton.setObjectName("clearButton")
        self.circleButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.circleButton.setGeometry(QtCore.QRect(20, 300, 141, 51))
        self.circleButton.setObjectName("circleButton")
        self.rectangleButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.rectangleButton.setGeometry(QtCore.QRect(20, 370, 141, 51))
        self.rectangleButton.setObjectName("rectangleButton")
        self.radius = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.radius.setGeometry(QtCore.QRect(630, 310, 101, 31))
        self.radius.setObjectName("radius")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(570, 310, 58, 31))
        self.label.setObjectName("label")
        self.centreX = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.centreX.setGeometry(QtCore.QRect(330, 310, 101, 31))
        self.centreX.setObjectName("centreX")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(190, 310, 58, 31))
        self.label_2.setObjectName("label_2")
        self.centreY = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.centreY.setGeometry(QtCore.QRect(440, 310, 101, 31))
        self.centreY.setObjectName("centreY")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(190, 380, 121, 31))
        self.label_3.setObjectName("label_3")
        self.bottomleftX = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.bottomleftX.setGeometry(QtCore.QRect(330, 380, 101, 31))
        self.bottomleftX.setObjectName("bottomleftX")
        self.bottomleftY = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.bottomleftY.setGeometry(QtCore.QRect(440, 380, 101, 31))
        self.bottomleftY.setObjectName("bottomleftY")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(190, 420, 131, 31))
        self.label_4.setObjectName("label_4")
        self.toprightX = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.toprightX.setGeometry(QtCore.QRect(330, 420, 101, 31))
        self.toprightX.setObjectName("toprightX")
        self.toprightY = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.toprightY.setGeometry(QtCore.QRect(440, 420, 101, 31))
        self.toprightY.setObjectName("toprightY")
        self.label_5 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(370, 280, 21, 31))
        self.label_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(480, 280, 21, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.openButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.openButton.setGeometry(QtCore.QRect(20, 470, 141, 51))
        self.openButton.setObjectName("openButton")
        self.saveButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(160, 470, 141, 51))
        self.saveButton.setObjectName("saveButton")
        self.label_7 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(480, 350, 21, 31))
        self.label_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(370, 350, 21, 31))
        self.label_8.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.line = QtWidgets.QFrame(parent=self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 340, 801, 31))
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(parent=self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(0, 450, 801, 31))
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(parent=self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(0, 270, 801, 31))
        self.line_3.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_3.setObjectName("line_3")
        self.widget = MplWidget(parent=self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 20, 781, 201))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DL_MESO_LBE Boundary Conditions"))
        self.wallsButton.setText(_translate("MainWindow", "Add walls"))
        self.clearButton.setText(_translate("MainWindow", "Clear all points"))
        self.circleButton.setText(_translate("MainWindow", "Add circle"))
        self.rectangleButton.setText(_translate("MainWindow", "Add rectangle"))
        self.label.setText(_translate("MainWindow", "Radius:"))
        self.label_2.setText(_translate("MainWindow", "Centre:"))
        self.label_3.setText(_translate("MainWindow", "Bottom left corner:"))
        self.label_4.setText(_translate("MainWindow", "Top right corner:"))
        self.label_5.setText(_translate("MainWindow", "x"))
        self.label_6.setText(_translate("MainWindow", "y"))
        self.openButton.setText(_translate("MainWindow", "Open lbin.spa"))
        self.saveButton.setText(_translate("MainWindow", "Save lbin.spa"))
        self.label_7.setText(_translate("MainWindow", "y"))
        self.label_8.setText(_translate("MainWindow", "x"))

# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        Canvas.updateGeometry(self)

# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.bcdata = np.zeros((50, 250), dtype=int)
        self.plot_bcdata()
        self.clearButton.clicked.connect(self.clear_data)
        self.openButton.clicked.connect(self.open_data)
        self.saveButton.clicked.connect(self.save_data)
        self.wallsButton.clicked.connect(self.add_walls)
        self.circleButton.clicked.connect(self.add_circle)
        self.rectangleButton.clicked.connect(self.add_rectangle)
    
    def clear_data(self):
        self.bcdata = np.zeros((50, 250), dtype=int)
        self.plot_bcdata()

    def open_data(self):
        bcdata = np.zeros((50, 250), dtype=int)
        try:
            with open('lbin.spa', 'r') as f:
                s = f.readlines()
            for line in s:
                words = line.split()
                if len(words)>3:
                    x = int(words[0])
                    y = int(words[1])
                    bc = int(words[3])
                    if bc>10 and x>=0 and x<250 and y>=0 and y<50:
                        bcdata[y][x] = 1
        except IOError:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msgbox.setWindowTitle("NO lbin.spa FILE FOUND")
            msgbox.setText('Cannot find an lbin.spa file or read its boundary conditions')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msgbox.exec()
        self.bcdata = bcdata
        self.plot_bcdata()

    def save_data(self):
        bc = ''
        for y in range(50):
            for x in range(250):
                if self.bcdata[y][x]==1:
                    # work out if boundary point is surrounded by other boundary points:
                    # if so, assign as blank site (11); if not, assign as mid-point bounceback site (13)
                    solid = (self.bcdata[(y+1)%50][x]==1 and self.bcdata[(y-1)%50][x]==1 and 
                            self.bcdata[y][(x+1)%250]==1 and self.bcdata[y][(x-1)%250]==1 and 
                            self.bcdata[(y+1)%50][(x+1)%250]==1 and self.bcdata[(y-1)%50][(x+1)%250]==1 and
                            self.bcdata[(y+1)%50][(x-1)%250]==1 and self.bcdata[(y-1)%50][(x-1)%250]==1)
                    if solid:
                        bc += "{0:d} {1:d} 0 11\n".format(x, y)
                    else:
                        bc += "{0:d} {1:d} 0 13\n".format(x, y)
        if os.path.isfile('lbin.spa'):
            msgbox = QtWidgets.QMessageBox()
            msgbox.setIcon(QtWidgets.QMessageBox.Icon.Question)
            msgbox.setWindowTitle("OVERWRITING lbin.spa?")
            msgbox.setText('An lbin.spa file already exists: do you wish to overwrite it?')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            button = msgbox.exec()
            if button == QtWidgets.QMessageBox.StandardButton.No:
                return
        with open('lbin.spa', 'w') as f:
            f.write(bc)
        msgbox = QtWidgets.QMessageBox()
        msgbox.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msgbox.setWindowTitle("WRITTEN lbin.spa FILE")
        msgbox.setText('Successfully written lbin.spa file with current boundary conditions')
        msgbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msgbox.exec()


    def add_walls(self):
        self.bcdata[0][0:250] = 1
        self.bcdata[49][0:250] = 1
        self.plot_bcdata()
    
    def add_circle(self):
        x = self.centreX.toPlainText()
        y = self.centreY.toPlainText()
        if x.isnumeric() and y.isnumeric():
            centreX = (int(x) % 250)
            centreY = (int(y) % 50)
        else:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msgbox.setWindowTitle("INVALID POSITION FOR CIRCLE")
            msgbox.setText('No position for centre of circle provided: cannot generate circle')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        r = self.radius.toPlainText()
        if r.isnumeric():
            circleR = abs(float(r))
            intR = round(circleR)
            if circleR>25:
                msgbox = QtWidgets.QMessageBox()
                msgbox.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                msgbox.setWindowTitle("INVALID RADIUS FOR CIRCLE")
                msgbox.setText('Radius for circle too large for system: cannot generate circle')
                msgbox.stooetStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
                msgbox.exec()
                return
        else:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msgbox.setWindowTitle("INVALID RADIUS FOR CIRCLE")
            msgbox.setText('No radius for circle provided: cannot generate circle')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        for yy in range(-intR, intR+1):
            yyy = (centreY+yy)%50
            for xx in range(-intR, intR+1):
                xxx = (centreX+xx)%250
                if xx*xx+yy*yy<circleR*circleR:
                    self.bcdata[yyy][xxx] = 1
        self.plot_bcdata()

    def add_rectangle(self):
        x1 = self.bottomleftX.toPlainText()
        y1 = self.bottomleftY.toPlainText()
        if x1.isnumeric() and y1.isnumeric():
            bottomleftX = (int(x1) % 250)
            bottomleftY = (int(y1) % 50)
        else:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msgbox.setWindowTitle("INVALID POSITION FOR RECTANGLE")
            msgbox.setText('No position for bottom-left of rectangle provided: cannot generate rectangle')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        x2 = self.toprightX.toPlainText()
        y2 = self.toprightY.toPlainText()
        if x2.isnumeric() and y2.isnumeric():
            toprightX = (int(x2) % 250)
            toprightY = (int(y2) % 50)
        else:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msgbox.setWindowTitle("INVALID POSITION FOR RECTANGLE")
            msgbox.setText('No position for top-right of rectangle provided: cannot generate rectangle')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        extentX = toprightX - bottomleftX
        if extentX<0:
            extentX = bottomleftX - toprightX
        extentY = toprightY - bottomleftY
        if extentY<0:
            extentY = bottomleftY - toprightY
        if extentX==0 or extentX>250 or extentY==0 or extentY>50:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msgbox.setWindowTitle("INVALID SIZE FOR RECTANGLE")
            msgbox.setText('Size of rectangle provided zero or too large for system: cannot generate rectangle')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        for yy in range(extentY):
            yyy = (min(bottomleftY, toprightY)+yy) % 50
            for xx in range(extentX):
                xxx = (min(bottomleftX, toprightX)+xx) % 250
                self.bcdata[yyy][xxx] = 1
        self.plot_bcdata()


    def plot_bcdata(self):
        my_cmap = mpl.colors.ListedColormap(['r', 'g', 'b'])
        my_cmap.set_bad(color='w', alpha=0)
        #for x in range(251):
        #    self.widget.canvas.ax.axvline(x, lw=0.25, color='k', zorder=5)
        #for y in range(51):
        #    self.widget.canvas.ax.axvline(y, lw=0.25, color='k', zorder=5)
        self.widget.canvas.ax.imshow(self.bcdata, interpolation='none', cmap=my_cmap, extent=[0, 250, 0, 50], origin='lower', zorder=0)
        #self.widget.canvas.ax.axis('off')
        self.widget.canvas.ax.xaxis.set_ticks(np.arange(0, 251, 25))
        self.widget.canvas.ax.yaxis.set_ticks(np.arange(0, 51, 10))
        self.widget.canvas.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

        
