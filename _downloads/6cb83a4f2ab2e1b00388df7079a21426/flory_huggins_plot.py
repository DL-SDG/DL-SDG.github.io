#!/usr/bin/env python3 
# -*- coding: UTF-8 -*-
"""Usage:
    flory_huggins_plot.py [--rho <rho>]

Plots data file for Flory-Huggins chi parameter determination, both concentration profiles obtained from simulations and chi parameters determined from profiles

Options:
    -h, --help          Display this message
    --rho <rho>         Particle density [default: 3.0]

michael.seaton@stfc.ac.uk, 22/07/22
"""

from docopt import docopt
import sys
from PyQt5.QtWidgets import QWidget,QApplication,QMainWindow,QLineEdit,\
        QPushButton,QLabel,QAction,QTableWidget,QTableWidgetItem,QVBoxLayout,\
        QHBoxLayout,QTabWidget,QGroupBox,QFormLayout,QComboBox,QSpinBox,\
        QDoubleSpinBox,QRadioButton,QSizePolicy,QCheckBox,QGridLayout,QMessageBox

from PyQt5.QtGui import QIcon,QIntValidator,QDoubleValidator
from PyQt5.QtCore import pyqtSlot, Qt
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
  QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
  QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import statistics
import math

class App(QMainWindow):
  def __init__(self):
    super().__init__()
    self.title="DL_MESO_DPD Flory-Huggins data plotter"
    self.left=0
    self.top=0
    s = app.primaryScreen().size()
    self.width=s.width()
    self.height=s.height()
    self.InitUI()

  def InitUI(self):
    self.setWindowTitle(self.title)
    self.setGeometry(self.left,self.top,self.width,self.height)
    self.statusBar().showMessage("earth, milky way")
    
    self.main = MyTabs(self)
    self.setCentralWidget(self.main)

    menu = self.menuBar()
    fileMenu = menu.addMenu('File')
    helpMenu = menu.addMenu('Help')

#  file menu buttons
    exit = QAction(QIcon('exit.png'), 'Exit', self)
    exit.setShortcut('Ctrl+Q')
    exit.setStatusTip('Exit application')
    exit.triggered.connect(self.close)
    fileMenu.addAction(exit)

    self.show()

class MyTabs(QWidget):
  def __init__(self,parent):
    super(QWidget, self).__init__(parent)

    self.layout = QVBoxLayout()

    self.tabs = QTabWidget() 
    self.tab_density = QWidget()
    self.tab_chi = QWidget()
    
    args = docopt(__doc__)
    filename = 'floryhuggins-rho-{0:.3f}.dat'.format(float(args["--rho"]))
    self.n,self.lx,self.data,self.datumNames,self.chi,self.chi_legend = readDensity(filename)

    self.tabs.addTab(self.tab_density,"Concentration")
    self.tab_density.layout = QVBoxLayout(self)
    self.plot = FigureCanvas(Figure(figsize=(16,12)))
    self.tab_density.layout.addWidget(NavigationToolbar(self.plot, self))
    self.tab_density.layout.addWidget(self.plot)
    self.tab_density.layout.addStretch(1)
    self.createDensitySeries(self.datumNames)
    self.tab_density.layout.addWidget(self.ds)
    self.ax = self.plot.figure.add_subplot(111)
    self.dsb.currentIndexChanged.connect(self.load_data)
    self.chileftmin.editingFinished.connect(self.update_data)
    self.chileftmax.editingFinished.connect(self.update_data)
    self.chirightmin.editingFinished.connect(self.update_data)
    self.chirightmax.editingFinished.connect(self.update_data)
    self.update_plot(self.dsb.currentIndex())
    self.chirecalc.clicked.connect(self.recalculate_chi)
    self.chireset.clicked.connect(self.reset_chi)
    self.tab_density.setLayout(self.tab_density.layout)

    self.tabs.addTab(self.tab_chi,"Mixing parameter")
    self.tab_chi.layout = QVBoxLayout(self)
    self.chi_plot = FigureCanvas(Figure(figsize=(16,12)))
    self.tab_chi.layout.addWidget(NavigationToolbar(self.chi_plot, self))
    self.tab_chi.layout.addWidget(self.chi_plot)
    self.tab_chi.layout.addStretch(1)
    self.axr = self.chi_plot.figure.add_subplot(111)
    self.createChiAnalysis()
    self.tab_chi.layout.addWidget(self.chian)
    self.update_chi()
    self.chi_minA.editingFinished.connect(self.do_chi_regress)
    self.chi_maxA.editingFinished.connect(self.do_chi_regress)
    self.tab_chi.setLayout(self.tab_chi.layout)

    self.layout.addWidget(self.tabs)
    self.setLayout(self.layout)

  def createDensitySeries(self,datumNames):
      self.ds = QGroupBox('Data')
      lt = QHBoxLayout()
      
      self.dsb = QComboBox()
      layout = QFormLayout()
      self.dsb.addItems(datumNames)
      self.dsb.setCurrentIndex(0)
      self.dsb.setLayout(layout)
  
      self.ds_chirecalc = QGroupBox("Recalculate chi")
      layout = QFormLayout()
      self.chileftmin = QLineEdit()
      self.chileftmin.setValidator(QDoubleValidator())
      self.chileftmin.setText('{0:f}'.format(self.lx[0]*0.15))
      layout.addRow(QLabel('xmin for left sample:'),self.chileftmin)
      self.chileftmax = QLineEdit()
      self.chileftmax.setValidator(QDoubleValidator())
      self.chileftmax.setText('{0:f}'.format(self.lx[0]*0.35))
      layout.addRow(QLabel('xmax for left sample:'),self.chileftmax)
      self.chirightmin = QLineEdit()
      self.chirightmin.setValidator(QDoubleValidator())
      self.chirightmin.setText('{0:f}'.format(self.lx[0]*0.65))
      layout.addRow(QLabel('xmin for right sample:'),self.chirightmin)
      self.chirightmax = QLineEdit()
      self.chirightmax.setValidator(QDoubleValidator())
      self.chirightmax.setText('{0:f}'.format(self.lx[0]*0.85))
      layout.addRow(QLabel('xmax for right sample:'),self.chirightmax)
      self.ds_chirecalc.setLayout(layout)
      self.chirecalc = QPushButton('recalculate', self)
      self.chireset = QPushButton('reset', self)
      layout.addRow(self.chirecalc, self.chireset)

      lt.addWidget(self.dsb)
      lt.addWidget(self.ds_chirecalc)
      self.ds.setLayout(lt)

  def createChiAnalysis(self):
      self.chian = QGroupBox("")
      layout = QHBoxLayout()
      self.do_chi_options()
      layout.addWidget(self.chi_options)
      self.chian.setLayout(layout)

  def load_data(self,c):
      cc = self.dsb.currentIndex()
      self.chileftmin.setText('{0:f}'.format(self.lx[cc]*0.15))
      self.chileftmax.setText('{0:f}'.format(self.lx[cc]*0.35))
      self.chirightmin.setText('{0:f}'.format(self.lx[cc]*0.65))
      self.chirightmax.setText('{0:f}'.format(self.lx[cc]*0.85))
      self.update_plot(self.dsb.currentIndex())

  
  def reset_chi(self,c):
      args = docopt(__doc__)
      filename = 'floryhuggins-rho-{0:.3f}.dat'.format(float(args["--rho"]))
      self.n,self.lx,self.data,self.datumNames,self.chi,self.chi_legend = readDensity(filename)
      cc = self.dsb.currentIndex()
      self.chileftmin.setText('{0:f}'.format(self.lx[cc]*0.15))
      self.chileftmax.setText('{0:f}'.format(self.lx[cc]*0.35))
      self.chirightmin.setText('{0:f}'.format(self.lx[cc]*0.65))
      self.chirightmax.setText('{0:f}'.format(self.lx[cc]*0.85))
      self.update_plot(cc)
      self.update_chi()

  def update_data(self):
      self.update_plot(self.dsb.currentIndex())
  
  def do_chi_regress(self):
      self.update_chi()

  def recalculate_chi(self):
      c=self.dsb.currentIndex()
      xmin1=float(self.chileftmin.text())
      xmax1=float(self.chileftmax.text())
      xmin2=float(self.chirightmin.text())
      xmax2=float(self.chirightmax.text())
      minchi = int(xmin1*len(self.data[2*c])/self.lx[c])
      maxchi = int(xmax1*len(self.data[2*c])/self.lx[c])
      meanvolfrac = statistics.mean(self.data[2*c+1][minchi:maxchi])
      stdvolfrac = statistics.stdev(self.data[2*c+1][minchi:maxchi])
      chi1 = math.log((1.0-meanvolfrac)/meanvolfrac)/(1.0-2.0*meanvolfrac)
      chimax1 = math.log((1.0-meanvolfrac-stdvolfrac)/(meanvolfrac+stdvolfrac))/(1.0-2.0*(meanvolfrac+stdvolfrac))
      chimin1 = math.log((1.0-meanvolfrac+stdvolfrac)/(meanvolfrac-stdvolfrac))/(1.0-2.0*(meanvolfrac-stdvolfrac))
      chierr1 = max(abs(chimax1-chi1), abs(chi1-chimin1))
      minchi = int(xmin2*len(self.data[2*c])/self.lx[c])
      maxchi = int(xmax2*len(self.data[2*c])/self.lx[c])
      meanvolfrac = statistics.mean(self.data[2*c+1][minchi:maxchi])
      stdvolfrac = statistics.stdev(self.data[2*c+1][minchi:maxchi])
      chi2 = math.log((1.0-meanvolfrac)/meanvolfrac)/(1.0-2.0*meanvolfrac)
      chimax2 = math.log((1.0-meanvolfrac-stdvolfrac)/(meanvolfrac+stdvolfrac))/(1.0-2.0*(meanvolfrac+stdvolfrac))
      chimin2 = math.log((1.0-meanvolfrac+stdvolfrac)/(meanvolfrac-stdvolfrac))/(1.0-2.0*(meanvolfrac-stdvolfrac))
      chierr2 = max(abs(chimax2-chi2), abs(chi2-chimin2))
      if(chierr2>chierr1):
        self.chi[c][2] = chi1
        self.chi[c][3] = chierr1
      else:
        self.chi[c][2] = chi2
        self.chi[c][3] = chierr2
      self.update_plot(self.dsb.currentIndex())
      self.update_chi()

  def update_chi(self):
    self.axr.clear()
    colourmap = cm.get_cmap('jet')
    nAii = len(self.chi_legend)
    fullset = []
    for i in range(nAii):
        fullset.append([x for x in self.chi if x[0]==self.chi_legend[i]])
    for i in range(nAii):
        tmpx = [(x[1]-x[0]) for x in fullset[i]]
        tmpy = [x[2] for x in fullset[i]]
        tmpe = [x[3] for x in fullset[i]]
        self.axr.scatter(tmpx, tmpy, 25, cmap=colourmap, marker='s', label='Aii = {0:s}'.format(str(self.chi_legend[i])), zorder=1)
        self.axr.errorbar(tmpx, tmpy, yerr=tmpe, linestyle='None', fmt='None', ecolor='gray', capsize=5, zorder=-1)
    xchi = [(x[1]-x[0]) for x in self.chi]
    chi = [x[2] for x in self.chi]
    dAmin = float(self.chi_minA.text())
    dAmax = float(self.chi_maxA.text())
    self.axr.set_xlabel('$\Delta$A')
    self.axr.set_ylabel('$\chi$')
    self.axr.set_xlim(left=0.0)
    self.axr.set_ylim(bottom=0.0)
    plotx = [0.0]
    xchidata = []
    chidata = []
    for i in range(len(chi)):
        x = xchi[i]
        plotx.append(x)
        if(xchi[i]>=dAmin and xchi[i]<=dAmax):
            xchidata.append(xchi[i])
            chidata.append(chi[i]/x)
    plotx = sorted(plotx)
    a = sum(chidata)/len(chidata)
    ra = 1.0/a
    amax = max(chidata)
    ramin = 1.0/amax
    amin = min(chidata)
    ramax = 1.0/amin
    aerr = max(amax-a, a-amin)
    raerr = max(ramax-ra, ra-ramin)
    fit_fn = np.poly1d([a, 0.0])
    self.axr.plot(plotx, fit_fn(plotx), '--k')
    self.axr.set_title('$\chi$ = ({0:f}$\pm${1:f})$\Delta$A\n$\Delta$A = ({2:f}$\pm${3:f})$\chi$'.format(a, aerr, ra, raerr))
    self.axr.legend()
    self.chi_plot.draw()

   
  def update_plot(self,c):
    self.ax.clear()
    self.ax.plot(self.data[2*c],self.data[2*c+1], 'r-')
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel('x [DPD length units]')
    self.ax.set_ylabel('Bead concentration, $\phi$')
    self.ax.set_xlim(left=min(self.data[2*c]),right=max(self.data[2*c]))
    self.ax.set_ylim(bottom=0.0, top=1.0)
    chilines=[float(self.chileftmin.text()), float(self.chileftmax.text()), float(self.chirightmin.text()), float(self.chirightmax.text())]
    for xc in chilines:
        self.ax.axvline(x=xc, color='k', linestyle='--')
    self.ax.legend(['$\chi$ = {0:f}$\pm${1:f}'.format(self.chi[c][2], self.chi[c][3])], loc='upper right')
    self.plot.draw()

  def do_chi_options(self):
    self.chi_options = QGroupBox('')
    lt = QHBoxLayout()
    self.chi_regress_options = QGroupBox("")
    layout = QFormLayout()
    self.chi_minA = QLineEdit()
    self.chi_minA.setValidator(QDoubleValidator())
    self.chi_minA.setText(str(10.0))
    layout.addRow(QLabel('Minimum value of dA for line-fitting: '),self.chi_minA)
    self.chi_maxA = QLineEdit()
    self.chi_maxA.setValidator(QDoubleValidator())
    self.chi_maxA.setText(str(100.0))
    layout.addRow(QLabel('Maximum value of dA for line-fitting: '),self.chi_maxA)
    self.chi_regress_options.setLayout(layout)
    lt.addWidget(self.chi_regress_options)
    self.chi_options.setLayout(lt)

def readDensity(filename):
  s = open(filename).read().split('\n')
  i = 0
  chidata = []
  datumNames = []
  Aiiset = []
  d = []
  l = []
  while i<len(s):
    line = s[i].split(',')
    if len(line)!=5:
        break
    Aii=float(line[0])
    Aij=float(line[1])
    chi=float(line[2])
    chisd=float(line[3])
    datumNames.append("Aii = {0:s}, Aij = {1:s}".format(str(Aii),str(Aij)))
    chidata.append([Aii, Aij, chi, chisd])
    datalength = int(line[4])
    Aiiset.append(Aii)
    pointdata = []
    ss = s[i+1:i+datalength+1]
    for line in ss:
        dataelement = [float(j) for j in line.split()]
        pointdata.append(dataelement)
    d.append([x[0] for x in pointdata])
    d.append([x[1] for x in pointdata])
    i += (datalength+1)

  legend = list(sorted(set(Aiiset)))
  n = len(datumNames)
  for i in range(n):
    dx = d[2*i][1]-d[2*i][0]
    maxlx = max(d[2*i][:])
    l.append(maxlx+0.5*dx)
  
  return n,l,d,datumNames,chidata,legend

if __name__ == '__main__':
  app = QApplication(sys.argv)
  exe = App()
  sys.exit(app.exec_())
