#!/usr/bin/env python3 
# -*- coding: UTF-8 -*-
"""Usage:
    dlmresultviewer.py [--cor <correl>] [--rdf <rdfdat>] [--msd <msddat>] [--loc <local>]

Plots and processes various outputs from DL_MESO_DPD (both generated during 
simulations and calculated afterwards in post-processing), including CORREL 
files with system-wide statistical properties, RDFDAT files with radial 
distribution functions, MSDDAT files with mean squared displacements, and 
structured grid VTK files (both legacy and XML-style formats) with localised 
properties calculated in voxels

Options:
    -h, --help          Show this screen
    --cor <correl>      Name of CORREL file with statistical properties [default: CORREL]
    --rdf <rdfdat>      Name of RDFDAT file with radial distribution functions
                        [default: RDFDAT]
    --msd <msddat>      Name of MSDDAT file with mean squared displacements 
                        [default: MSDDAT]
    --loc <localvtk>    Name of structured grid VTK file with localised properties 
                        (filename extension can either be .vtk for legacy format or .vts
                        for XML-style format) [default: averages.vtk]

michael.seaton@stfc.ac.uk, 23/06/23
"""

import sys
from PyQt5.QtWidgets import QWidget,QApplication,QMainWindow,QLineEdit,\
        QPushButton,QLabel,QAction,QTableWidget,QTableWidgetItem,QVBoxLayout,\
        QHBoxLayout,QTabWidget,QGroupBox,QFormLayout,QComboBox,QSpinBox,QSlider,\
        QDoubleSpinBox,QRadioButton,QSizePolicy,QCheckBox,QGridLayout,QMessageBox,QFileDialog

from PyQt5.QtGui import QIcon,QIntValidator,QDoubleValidator
from PyQt5.QtCore import pyqtSlot, Qt
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
  QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
  QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from docopt import docopt
from matplotlib.figure import Figure
import numpy as np
from numpy.polynomial import Polynomial
import math
import sys,os
from scipy.stats import gaussian_kde
from scipy.fft import dst
import vtk
from vtk.util import numpy_support as VN    

class App(QMainWindow):
  def __init__(self):
    super().__init__()
    self.title="DL_MESO_DPD output file visualiser"
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
    #self.createActions()
    #self.main.layout.addWidget(self.act)

    menu = self.menuBar()
    fileMenu = menu.addMenu('File')
    helpMenu = menu.addMenu('Help')

#  file menu buttons

    saveLocal = QAction(QIcon('save.png'), 'Save localised data', self)
    saveLocal.setShortcut('Ctrl+S')
    saveLocal.setStatusTip('Save localised data as tabulated text file')
    saveLocal.triggered.connect(self.save_local)
    fileMenu.addAction(saveLocal)

    exit = QAction(QIcon('exit.png'), 'Exit', self)
    exit.setShortcut('Ctrl+Q')
    exit.setStatusTip('Exit application')
    exit.triggered.connect(self.close)
    fileMenu.addAction(exit)

    self.show()

  def save_local(self):
    args = docopt(__doc__)
    localvtk = args["--loc"]
    if localvtk[-4:]==".vtk" or localvtk[-4:]==".vts":
       localvtk = localvtk[:-4]
    nx,ny,nz,nd,local_data,local_labels,local_shortlabels,local_datumAxes=readLocal(localvtk)
    if nd>0:
      filename, _ = QFileDialog.getSaveFileName(self, 'Save localised data')
      line = self.main.localline.currentIndex()
      ix = self.main.localx.value()
      iy = self.main.localy.value()
      iz = self.main.localz.value()
      so = "#           position"
      for i in range(nd):
         so += "   {:>17}".format(local_shortlabels[i])
      so += "\n"
      if line==0:
        for i in range(len(local_data[0][:,iy,iz])):
          so += "{0:20.6f}".format(local_data[0][i,iy,iz])
          for j in range(nd):
            if local_shortlabels[j].startswith('bead_numbers'):
              so += "{0:20d}".format(local_data[j+3][i,iy,iz])
            else:
              so += "{0:20.6f}".format(local_data[j+3][i,iy,iz])
          so += "\n"
      elif line==1:
        for i in range(len(local_data[1][ix,:,iz])):
          so += "{0:20.6f}".format(local_data[1][ix,i,iz])
          for j in range(nd):
            if local_shortlabels[j].startswith('bead_numbers'):
              so += "{0:20d}".format(local_data[j+3][ix,i,iz])
            else:
              so += "{0:20.6f}".format(local_data[j+3][ix,i,iz])
          so += "\n"
      else:
        for i in range(len(local_data[2][ix,iy,:])):
          so += "{0:20.6f}".format(local_data[2][ix,iy,i])
          for j in range(nd):
            if local_shortlabels[j].startswith('bead_numbers'):
              so += "{0:20d}".format(local_data[j+3][ix,iy,i])
            else:
              so += "{0:20.6f}".format(local_data[j+3][ix,iy,i])
          so += "\n"
      open(filename,"w").write(so)
    else:
      msgbox = QMessageBox()
      msgbox.setIcon(QMessageBox.Warning)
      msgbox.setWindowTitle('WARNING')
      msgbox.setText('No localised data available to save')
      msgbox.setStandardButtons(QMessageBox.Ok)
      msgbox.exec_()



class MyTabs(QWidget):
  def __init__(self,parent):
    # get names of files to open (removing extension for VTK file)
    args = docopt(__doc__)
    correl = args["--cor"]
    rdfdat = args["--rdf"]
    msddat = args["--msd"]
    localvtk = args["--loc"]

    if localvtk[-4:]==".vtk" or localvtk[-4:]==".vts":
       localvtk = localvtk[:-4]

    super(QWidget, self).__init__(parent)

    self.layout = QVBoxLayout()

    self.tabs = QTabWidget() 
    self.tab_correl = QWidget()
    self.tab_rdf = QWidget()
    self.tab_msd = QWidget()
    self.tab_local = QWidget()
    #self.tab_advanced = QWidget()

    numtab = 0

    self.n,self.nd,self.data,self.datumNames,self.datumAxes = readCorrel(correl)
    if self.n>0:
      self.tabs.addTab(self.tab_correl,"Properties")
      self.tab_correl.layout = QVBoxLayout(self)
      self.plot = FigureCanvas(Figure(figsize=(16,12)))
      self.tab_correl.layout.addWidget(NavigationToolbar(self.plot, self))
      self.tab_correl.layout.addWidget(self.plot)
      self.tab_correl.layout.addStretch(1)
      self.createTimeSeries(self.datumNames)
      self.tab_correl.layout.addWidget(self.ts)
      self.ax = self.plot.figure.add_subplot(111)
      self.tsb.currentIndexChanged.connect(self.load_data)
      self.tsb.currentIndexChanged.connect(self.update_limits)
      self.update_plot(self.tsb.currentIndex())
      self.createAnalysis()
      self.tab_correl.layout.addWidget(self.an)
      self.ana.currentIndexChanged.connect(self.do_analysis)
      self.bins.editingFinished.connect(self.do_histogram)
      self.pdf.stateChanged.connect(self.do_norm)
      self.lag.editingFinished.connect(self.do_autocorr)
      self.nsample.editingFinished.connect(self.do_run_ave)
      self.xmin.editingFinished.connect(self.do_run_ave)
      self.xmax.editingFinished.connect(self.do_run_ave)
      self.tab_correl.setLayout(self.tab_correl.layout)
      numtab += 1

    self.nrdf,self.nprdf,self.rdf_data,self.rdf_labels=readRDF(rdfdat)
    if self.nrdf>0 :
      self.tabs.addTab(self.tab_rdf,"RDFs")
      self.tab_rdf.layout = QVBoxLayout(self)
      self.rdf_plot = FigureCanvas(Figure(figsize=(16,12)))
      self.tab_rdf.layout.addWidget(NavigationToolbar(self.rdf_plot, self))
      self.tab_rdf.layout.addWidget(self.rdf_plot)
      self.tab_rdf.layout.addStretch(1) 
      self.axr = self.rdf_plot.figure.add_subplot(111)
      self.createRDFS(self.rdf_labels)
      self.createRDFAnalysis()
      self.tab_rdf.layout.addWidget(self.rg)
      self.tab_rdf.layout.addWidget(self.rdfan)
      self.tab_rdf.layout.addStretch(1)
      self.rdfs.currentIndexChanged.connect(self.load_rdf)
      self.update_rdf(self.rdfs.currentIndex())
      self.rdf_ana.currentIndexChanged.connect(self.load_rdf)
      self.rdf_ftsize.editingFinished.connect(self.do_rdf_ft)
      self.tab_rdf.setLayout(self.tab_rdf.layout)
      numtab += 1

    self.nmsd,self.npmsd,self.msd_data,self.msd_labels=readMSD(msddat)
    if self.nmsd>0 :
      self.tabs.addTab(self.tab_msd,"MSDs")
      self.tab_msd.layout = QVBoxLayout(self)
      self.msd_plot = FigureCanvas(Figure(figsize=(16,12)))
      self.tab_msd.layout.addWidget(NavigationToolbar(self.msd_plot, self))
      self.tab_msd.layout.addWidget(self.msd_plot)
      self.tab_msd.layout.addStretch(1)
      self.axm = self.msd_plot.figure.add_subplot(111)
      self.createMSDS(self.msd_labels)
      self.createMSDAnalysis()
      self.tab_msd.layout.addWidget(self.rmsd)
      self.tab_msd.layout.addWidget(self.msdan)
      self.tab_msd.layout.addStretch(1)
      self.msds.currentIndexChanged.connect(self.load_msd)
      self.update_msd(self.msds.currentIndex())
      self.msd_ana.currentIndexChanged.connect(self.load_msd)
      self.msd_gradsize.editingFinished.connect(self.do_msd_grad)
      self.msdhist.stateChanged.connect(self.do_msd_grad)
      self.msdbins.editingFinished.connect(self.do_msd_grad)
      self.msdpdf.stateChanged.connect(self.do_msd_grad)
      self.tab_msd.setLayout(self.tab_msd.layout)
      numtab += 1

    self.nxlocal,self.nylocal,self.nzlocal,self.ndlocal,self.local_data,self.local_labels,self.local_shortlabels,self.local_datumAxes=readLocal(filename="averages")
    if self.ndlocal>0:
      self.tabs.addTab(self.tab_local,"Localised data")
      self.tab_local.layout = QVBoxLayout(self)
      self.local_plot = FigureCanvas(Figure(figsize=(16,12)))
      self.tab_local.layout.addWidget(NavigationToolbar(self.local_plot, self))
      self.tab_local.layout.addWidget(self.local_plot)
      self.tab_local.layout.addStretch(1)
      self.axl = self.local_plot.figure.add_subplot(111)
      if self.nxlocal>=self.nylocal and self.nxlocal>=self.nzlocal:
         line = 0
         ix = 0
         iy = self.nylocal//2
         iz = self.nzlocal//2
      elif self.nylocal>=self.nxlocal and self.nylocal>=self.nzlocal:
         line = 1
         ix = self.nxlocal//2
         iy = 0
         iz = self.nzlocal//2
      else:
         line = 2
         ix = self.nxlocal//2
         iy = self.nylocal//2
         iz = 0
      self.createLocalS(self.local_labels,line,self.nxlocal,self.nylocal,self.nzlocal)
      self.createLocalAnalysis()
      bottomrow = QHBoxLayout()
      bottomrow.addWidget(self.rlocal)
      bottomrow.addWidget(self.localan)
      self.tab_local.layout.addLayout(bottomrow)
      self.tab_local.layout.addStretch(1)
      self.locals.currentIndexChanged.connect(self.load_local)
      self.localline.currentIndexChanged.connect(self.load_local)
      self.localx.valueChanged.connect(self.load_local)
      self.localy.valueChanged.connect(self.load_local)
      self.localz.valueChanged.connect(self.load_local)
      self.update_local(self.locals.currentIndex(),line,ix,iy,iz)
      self.local_ana.currentIndexChanged.connect(self.load_local)
      self.locallineplot.stateChanged.connect(self.load_local)
      self.localregorder.currentIndexChanged.connect(self.load_local)
      self.localregeqn.stateChanged.connect(self.load_local)
      self.tab_local.setLayout(self.tab_local.layout)
      numtab += 1

    if numtab > 0:
      self.layout.addWidget(self.tabs)
      self.setLayout(self.layout)
    else:
       # print error message: no data can be found
       sys.exit("ERROR: cannot find any DL_MESO_DPD simulation data to plot - exiting now")

  def createRDFS(self,rdfnames):
    layout=QHBoxLayout()
    self.rg=QGroupBox("Data")
    self.rdfs = QComboBox()
    self.rdfs.addItems(rdfnames)
    self.rdfs.setCurrentIndex(0)
    label=QLabel("Data set")
    layout.addWidget(label)
    layout.addWidget(self.rdfs)
    layout.addStretch(1)
    self.rg.setLayout(layout)

  def createRDFAnalysis(self):
      self.rdfan = QGroupBox("Toolbox")
      layout = QHBoxLayout()
      self.rdf_ana = QComboBox()
      self.rdf_ana.addItems(['Radial Distribution Function','Fourier Transform'])
      self.rdf_ana.setCurrentIndex(0)
      label = QLabel("Analysis:")
      layout.addWidget(label)
      layout.addWidget(self.rdf_ana)
      self.do_rdf_options()
      layout.addWidget(self.rdf_tool_options)
      self.rdfan.setLayout(layout)

  def createMSDS(self,msdnames):
    layout=QHBoxLayout()
    self.rmsd=QGroupBox("Data")
    self.msds = QComboBox()
    self.msds.addItems(msdnames)
    self.msds.setCurrentIndex(0)
    label=QLabel("Data set")
    layout.addWidget(label)
    layout.addWidget(self.msds)
    layout.addStretch(1)
    self.rmsd.setLayout(layout)

  def createMSDAnalysis(self):
      self.msdan = QGroupBox("Toolbox")
      layout = QHBoxLayout()
      self.msd_ana = QComboBox()
      self.msd_ana.addItems(['Mean Squared Displacement','Diffusivity'])
      self.msd_ana.setCurrentIndex(0)
      label = QLabel("Analysis:")
      layout.addWidget(label)
      layout.addWidget(self.msd_ana)
      self.do_msd_options()
      layout.addWidget(self.msd_tool_options)
      self.msdan.setLayout(layout)
    
  def createLocalS(self,localnames,line,nx,ny,nz):
    layout=QVBoxLayout()
    self.rlocal=QGroupBox("Data")
    self.locals = QComboBox()
    self.locals.addItems(localnames)
    self.locals.setCurrentIndex(0)
    label=QLabel("Data set")
    layout.addWidget(label)
    layout.addWidget(self.locals)
    self.localline = QComboBox()
    self.localline.addItems(['x','y','z'])
    self.localline.setCurrentIndex(line)
    label=QLabel("Direction of line")
    layout.addWidget(label)
    layout.addWidget(self.localline)
    self.localx = QSlider(Qt.Horizontal)
    self.localx.setMinimum(0 if nx>1 else -1)
    self.localx.setMaximum(nx-1 if nx>1 else 1)
    self.localx.setValue(nx//2)
    self.localx.setEnabled(nx>1)
    self.localy = QSlider(Qt.Horizontal)
    self.localy.setMinimum(0 if ny>1 else -1)
    self.localy.setMaximum(ny-1 if ny>1 else 1)
    self.localy.setValue(ny//2)
    self.localy.setEnabled(ny>1)
    self.localz = QSlider(Qt.Horizontal)
    self.localz.setMinimum(0 if nz>1 else -1)
    self.localz.setMaximum(nz-1 if nz>1 else 1)
    self.localz.setValue(nz//2)
    self.localz.setEnabled(nz>1)
    label=QLabel("Orthogonal x-position")
    layout.addWidget(label)
    layout.addWidget(self.localx)
    label=QLabel("Orthogonal y-position")
    layout.addWidget(label)
    layout.addWidget(self.localy)
    label=QLabel("Orthogonal z-position")
    layout.addWidget(label)
    layout.addWidget(self.localz)
    layout.addStretch(1)
    self.rlocal.setLayout(layout)

  def createLocalAnalysis(self):
      self.localan = QGroupBox("Toolbox")
      layout = QHBoxLayout()
      self.local_ana = QComboBox()
      self.local_ana.addItems(['Data only','Mean value','Linear regression'])
      self.local_ana.setCurrentIndex(0)
      label = QLabel("Analysis:")
      layout.addWidget(label)
      layout.addWidget(self.local_ana)
      self.do_local_options()
      layout.addWidget(self.local_tool_options)
      self.localan.setLayout(layout)

  def createTimeSeries(self,datumNames):
      self.ts = QGroupBox("Data")
      layout = QHBoxLayout()
      self.tsb = QComboBox()
      self.tsb.addItems(datumNames)
      self.tsb.setCurrentIndex(0)
      label = QLabel("Data set")
      layout.addWidget(label)
      layout.addWidget(self.tsb)
      layout.addStretch(1)
      self.ts.setLayout(layout)

  def createAnalysis(self):
      self.an = QGroupBox("Toolbox")
      layout = QHBoxLayout()
      self.ana = QComboBox()
      self.ana.addItems(['Timeseries','Histogram','Running average','Autocorrelation','Fourier Transform'])
      self.ana.setCurrentIndex(0)
      label = QLabel("Analysis:")
      layout.addWidget(label)
      layout.addWidget(self.ana)
      self.do_options()
      layout.addWidget(self.tool_options)
      self.an.setLayout(layout)

  def do_analysis(self,c):
      self.toggle_ana_options(c)
      if c == 0:
          self.update_plot(self.tsb.currentIndex())
      elif c == 1:    
          self.do_histogram(self.tsb.currentIndex())
      elif c == 2:    
          self.do_run_ave(self.tsb.currentIndex())
      elif c == 3:    
          self.do_autocorr(self.tsb.currentIndex())
      elif c == 4:    
          self.do_ft(self.tsb.currentIndex())

  def load_data(self,c):
      self.do_analysis(self.ana.currentIndex())

  def do_rdf_analysis(self,c):
      self.toggle_rdf_ana_options(c)
      if c == 0:
          self.update_rdf(self.rdfs.currentIndex())
      elif c == 1:
          self.do_rdf_ft(self.rdfs.currentIndex())

  def load_rdf(self,c):
    self.do_rdf_analysis(self.rdf_ana.currentIndex())

  def update_rdf(self,c):
    self.axr.clear()
    self.axr.plot(self.rdf_data[c,:,0],self.rdf_data[c,:,1], 'r-')
    self.axr.set_title(self.rdf_labels[c])
    self.axr.set_xlabel('r [DPD length units]')
    self.axr.set_ylabel('g(r)')
    self.axr.set_xlim(left=0.0,right=max(self.rdf_data[c,:,0]))
    self.rdf_plot.draw()

  def do_rdf_ft(self,c=-1):
    c= self.rdfs.currentIndex()
    self.axr.clear()
    f = int(self.rdf_ftsize.text())
    if f<self.nprdf:
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle('WARNING')
        msgbox.setText('FT bin size out of range: must be at least {0:d}'.format(self.nprdf))
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()
        return
    # set up data array with additional (padding) zeros beyond number of RDF points
    ft = np.zeros(f,dtype=float)
    ft[0:self.nprdf] = self.rdf_data[c,0:self.nprdf,1]-1.0
    # shift data to match points with frequencies
    for i in range(f-1):
      ft[i] = 0.5*(ft[i]+ft[i+1])
    ft[-1] = 0.0
    d=self.rdf_data[c,1,0]-self.rdf_data[c,0,0]
    freq = np.linspace(1.0,float(f),f)
    ft = dst(d*(freq-0.5)*ft)
    self.axr.set_xlabel('Reciprocal distance, $r^{-1}$ [$\ell_0^{-1}$]')
    self.axr.set_ylabel('FT of g(r), '+r'$S \left(r^{-1}\right)$')
    self.axr.set_title("Structure factor of "+self.rdf_labels[c])
    self.axr.plot(math.pi/(d*float(f))*freq,1.0+2.0*float(f)*d*d*ft/freq,'g-')
    self.rdf_plot.draw()

  def do_msd_analysis(self,c):
      self.toggle_msd_ana_options(c)
      if c == 0:
          self.update_msd(self.msds.currentIndex())
      elif c == 1:
          self.do_msd_grad(self.msds.currentIndex())

  def load_msd(self,c):
    self.do_msd_analysis(self.msd_ana.currentIndex())

  def update_msd(self,c):
    self.axm.clear()
    self.axm.plot(self.msd_data[c,:,0],self.msd_data[c,:,1], 'r-')
    self.axm.set_title(self.msd_labels[c])
    self.axm.set_xlabel('Time, t ['+r'$\tau_0$'+']')
    self.axm.set_ylabel('MSD(t) ['+r'$\ell_0^2 \tau_0^{-1}$'+']')
    self.axm.set_xlim(left=0.0,right=max(self.msd_data[c,:,0]))
    self.msd_plot.draw()

  def load_local(self,c):
      self.do_local_analysis(self.local_ana.currentIndex())

  def update_local(self,c,line,ix,iy,iz):
    self.axl.clear()
    if line==0:
        if self.locallineplot.isChecked() and self.nxlocal>1:
            self.axl.plot(self.local_data[line][:,iy,iz],self.local_data[c+3][:,iy,iz], 'r-')
        else:
            self.axl.plot(self.local_data[line][:,iy,iz],self.local_data[c+3][:,iy,iz], 'r.')
        self.axl.set_title(self.local_labels[c]+': y = {0:f}, z = {1:f}'.format(self.local_data[1][0,iy,iz], self.local_data[2][0,iy,iz]))
        s = np.mean(self.local_data[c+3][:,iy,iz])
        se = np.std(self.local_data[c+3][:,iy,iz])
    elif line==1:
        if self.locallineplot.isChecked() and self.nylocal>1:
            self.axl.plot(self.local_data[line][ix,:,iz],self.local_data[c+3][ix,:,iz], 'r-')
        else:
            self.axl.plot(self.local_data[line][ix,:,iz],self.local_data[c+3][ix,:,iz], 'r.')
        self.axl.set_title(self.local_labels[c]+': x = {0:f}, z = {1:f}'.format(self.local_data[0][ix,0,iz], self.local_data[2][ix,0,iz]))
        s = np.mean(self.local_data[c+3][ix,:,iz])
        se = np.std(self.local_data[c+3][ix,:,iz])
    elif line==2:
        if self.locallineplot.isChecked() and self.nzlocal>1:
            self.axl.plot(self.local_data[line][ix,iy,:],self.local_data[c+3][ix,iy,:], 'r-')
        else:
            self.axl.plot(self.local_data[line][ix,iy,:],self.local_data[c+3][ix,iy,:], 'r.')
        self.axl.set_title(self.local_labels[c]+': x = {0:f}, y = {1:f}'.format(self.local_data[0][ix,iy,0], self.local_data[1][ix,iy,0]))
        s = np.mean(self.local_data[c+3][ix,iy,:])
        se = np.std(self.local_data[c+3][ix,iy,:])
    self.localave.setText("{0:.8E} ± {1:.8E}".format(s,se))
    self.axl.set_ylabel(self.local_datumAxes[c])
    mx = self.local_data[line][0,0,0] + self.local_data[line][self.nxlocal-1,self.nylocal-1,self.nzlocal-1]
    if line==0:
        self.axl.set_xlabel('Position, '+r'$x$')
    elif line==1:
        self.axl.set_xlabel('Position, '+r'$y$')       
    else:
        self.axl.set_xlabel('Position, '+r'$z$')
    self.axl.set_xlim(left=0.0,right=mx)
    self.local_plot.draw()

  def do_local_analysis(self,c):
      line = self.localline.currentIndex()
      ix = self.localx.value()
      iy = self.localy.value()
      iz = self.localz.value()
      if line==0:
        s = np.mean(self.local_data[self.locals.currentIndex()+3][:,iy,iz])
        se = np.std(self.local_data[self.locals.currentIndex()+3][:,iy,iz])
      elif line==1:
        s = np.mean(self.local_data[self.locals.currentIndex()+3][ix,:,iz])
        se = np.std(self.local_data[self.locals.currentIndex()+3][ix,:,iz])
      elif line==2:
        s = np.mean(self.local_data[self.locals.currentIndex()+3][ix,iy,:])
        se = np.std(self.local_data[self.locals.currentIndex()+3][ix,iy,:])
      self.localave.setText("{0:.8E} ± {1:.8E}".format(s,se))
      if c == 0:
          self.update_local(self.locals.currentIndex(),line,ix,iy,iz)
      elif c == 1:
          self.do_local_ave(self.locals.currentIndex(),line,ix,iy,iz)
      elif c == 2:
          self.do_local_regress(self.locals.currentIndex(),line,ix,iy,iz)


  def do_msd_grad(self,c=-1):
    c= self.msds.currentIndex()
    self.axm.clear()
    f = int(self.msd_gradsize.text())
    if f>self.npmsd or f<2:
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle('WARNING')
        msgbox.setText('Gradient block-averaging bin size out of range: must be at least 2 and no more than {0:d}'.format(self.npmsd))
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()
        return
    dt = self.msd_data[c,1,0] - self.msd_data[c,0,0]
    numblock = self.npmsd-f+1
    grad = np.zeros((numblock,2),dtype=float)
    for i in range(numblock):
        grad[i][1] = (self.msd_data[c,i+f-1,1] - self.msd_data[c,i,1]) / (6.0*dt*float(f-1))
        grad[i][0] = 0.5*(self.msd_data[c,i,0]+self.msd_data[c,i+f-1,0])
    avgrad = np.mean(grad[:,1])
    sdgrad = np.std(grad[:,1])
    errortext = '±{0:f}'.format(sdgrad) if numblock>1 else ''
    self.axm.set_title('Self-diffusivity of {0:s} = {1:f}'.format(self.msd_labels[c],avgrad) + errortext)
    if self.msdhist.isChecked():
        self.axm.set_xlabel('D ['+r'$\ell_0^2 \tau_0^{-1}$'+']')
        self.axm.set_ylabel('Probability density function')
        n, x, _ = self.axm.hist(grad[:,1], bins=int(self.msdbins.text()),histtype='bar',rwidth=0.9,density=self.msdpdf.isChecked())
        dens = gaussian_kde(grad[:,1])
        self.axm.plot(x, dens(x))
    else:
        self.axm.set_xlabel('Time, t ['+r'$\tau_0$'+']')
        self.axm.set_ylabel('D(t) ['+r'$\ell_0^2 \tau_0^{-1}$'+']')
        self.axm.set_title('Self-diffusivity of {0:s} = {1:f}'.format(self.msd_labels[c],avgrad) + errortext)
        self.axm.plot(grad[:,0],grad[:,1],'g-')
    self.msd_plot.draw()

   
   
  def update_plot(self,c):
    self.ax.clear()
    self.ax.plot(self.data[:,0],self.data[:,c+1], 'r-')
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel('Time, t ['+r'$\tau_0$'+']')
    self.ax.set_ylabel(self.datumAxes[c])
    self.ax.set_xlim(left=min(self.data[:,0]),right=max(self.data[:,0]))
    self.plot.draw()

  def do_run_ave(self,c=-1):
    c= self.tsb.currentIndex()
    self.ax.clear()
    xmin=float(self.xmin.text())
    xmax=float(self.xmax.text())
    y = self.data[:,0]
    inx=np.intersect1d(np.where(y>=xmin),np.where(y<=xmax))[::int(self.nsample.text())]
    x =np.empty([len(inx)],dtype=float)
    y =np.empty([len(inx)],dtype=float)
    v =np.empty([len(inx)],dtype=float)
    s=0.0
    k=0
    for i in inx:
      k=k+1
      s = s + self.data[i,c+1]
      y[k-1] = self.data[i,c+1]
      v[k-1] = s/k
      x[k-1] = self.data[i,0]
    av = v[k-1] 
    ic=self.error.currentIndex()
    se=0.0
    if ic == 0:
      se = np.std(y)
    elif ic == 1:
      w=int(self.nwin.text())
      nb=len(y)//w
      if nb<2:
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle('WARNING')
        msgbox.setText('Insufficient samples for error calculation: decrease window size or increase data range for sampling')
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()
        return
      else:
        avj=[np.average(y[i*w:(i+1)*w]) for i in range(nb)]
        var=np.sum((avj-av)*(avj-av))
        se = np.sqrt(var/nb/(nb-1))
    elif ic == 2:
      w=int(self.nwin.text())
      n=len(y)
      nb=n//w
      if nb<1:
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle('WARNING')
        msgbox.setText('Insufficient samples for error calculation: decrease window size or increase data range for sampling')
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()
        return
      else:
        avj=[np.average(y[i*w:(i+1)*w]) for i in range(nb)]
        nav=[ (n*av-w*avj[i])/(n-w) for i in range(nb) ]
        var=np.sum((nav-av)*(nav-av))
        se = np.sqrt(var/nb*(nb-1))


    self.ave.setText("{0:.8E} ± {1:.8E}".format(v[k-1],se))
    self.ax.plot(self.data[:,0],self.data[:,c+1], 'r-',x,v,'b-')
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel('Time, t ['+r'$\tau_0$'+']')
    self.ax.set_ylabel(self.datumAxes[c])
    self.ax.set_xlim(left=min(self.data[:,0]),right=max(self.data[:,0]))
    self.plot.draw()


  def do_autocorr(self,l=-1):
    self.ax.clear()
    l = int(self.lag.text())
    if l<1 or l>self.n:
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle('WARNING')
        msgbox.setText('Time lag out of range: must be between 1 and {0:d}'.format(self.n))
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()
        return
    c= self.tsb.currentIndex()
    av = np.average(self.data[:,c+1])
    var = np.var(self.data[:,c+1])
    y=self.data[:,c+1]-av
    n=len(y)
    dt=self.data[1,0]-self.data[0,0]
    x=np.arange(0,float(l)*dt,dt)
    ci=np.correlate(y,y,mode='full')[-n:]
    ci=np.array([(y[:n-k]*y[-(n-k):]).sum() for k in range(l)])
    ci=ci/(var*(np.arange(n,n-l,-1)-x))
    
    self.ax.plot(x,ci, 'r-')
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel('Time lag, $t - t_0$ ['+r'$\tau_0$'+']')
    self.ax.set_ylabel('Autocorrelation function, '+r'$\langle f(t) f(t_0) \rangle / \langle f(t_0) f(t_0) \rangle$')
    self.ax.set_xlim(left=0,right=l)
    self.plot.draw()

  def do_norm(self,c):
      self.do_histogram()
  
  def do_histogram(self,c=-1):
    if c == -1 :
        c= self.tsb.currentIndex()
    self.ax.clear()
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel(self.datumAxes[c])
    self.ax.set_ylabel('Probability density function')
    n, x, _ = self.ax.hist(self.data[:,c+1], bins=int(self.bins.text()),histtype='bar',rwidth=0.9,density=self.pdf.isChecked())
    dens = gaussian_kde(self.data[:,c+1])
    self.ax.plot(x, dens(x))
    self.plot.draw()

  def do_ft(self,c):
    self.ax.clear()
    ft = np.fft.rfft(self.data[:,c+1])
    d=self.data[1,0]-self.data[0,0]
    freq = np.fft.rfftfreq(self.n, d=d)
    self.ax.set_xlabel('Frequency, $f$ ['+r'$\tau_0^{-1}$'+']')
    self.ax.set_ylabel('Fourier transform')
    self.ax.set_title("FT - "+self.datumNames[c])
    self.ax.plot(freq,2*np.abs(ft)/self.n,'g-')
    self.plot.draw()

  def update_limits(self,c):
    self.xmax.setText('{0:10.4f}'.format(self.data[-1,0]))
    self.xmin.setText('{0:10.4f}'.format(self.data[0,0]))

  def do_options(self):    
    self.tool_options = QGroupBox('Options')
    lt = QHBoxLayout()

    self.hist_options = QGroupBox("Histogram")
    layout = QFormLayout()
    self.bins = QLineEdit()
    self.bins.setValidator(QIntValidator())
    self.bins.setText('42')
    layout.addRow(QLabel('Bins '),self.bins)
    self.pdf = QCheckBox()
    layout.addRow(QLabel('Normalised? '),self.pdf)
    self.hist_options.setLayout(layout)


    self.ave_options = QGroupBox("Average")
    layout = QFormLayout()
    self.nsample = QLineEdit()
    self.nsample.setValidator(QIntValidator())
    self.nsample.setText('1')
    layout.addRow(QLabel('Sample '),self.nsample)
    self.xmin = QLineEdit()
    self.xmin.setValidator(QDoubleValidator())
    self.xmin.setText('{0:10.4f}'.format(self.data[0][0]))
    layout.addRow(QLabel('xmin'),self.xmin)
    self.xmax = QLineEdit()
    self.xmax.setValidator(QDoubleValidator())
    self.xmax.setText('{0:10.4f}'.format(self.data[self.n-1][0]))
    layout.addRow(QLabel('xmax'),self.xmax)
    self.error = QComboBox()
    self.error.addItems(['std error','binning','jackknife'])
    self.error.setCurrentIndex(0)
    self.error.currentIndexChanged.connect(self.do_error)
    lh=QHBoxLayout()
    lh.addWidget(QLabel('Error'))
    lh.addWidget(self.error)
    lh.addWidget(QLabel('Window size: '))
    self.nwin = QLineEdit()
    self.nwin.setValidator(QIntValidator())
    self.nwin.setText('100')
    self.nwin.setEnabled(False)
    lh.addWidget(self.nwin)
    layout.addRow(lh)
    self.ave =QLabel()
    layout.addRow(QLabel('Average:  '),self.ave)
    self.ave_options.setLayout(layout)

    self.auto_options = QGroupBox("AutoCorrelation")
    layout = QFormLayout()
    self.lag = QLineEdit()
    self.lag.setValidator(QIntValidator())
    self.lag.setText(str(min(self.n//2, 250)))
    layout.addRow(QLabel('Lag: '),self.lag)
    self.auto_options.setLayout(layout)


    self.toggle_ana_options(0)
    lt.addWidget(self.hist_options)
    lt.addWidget(self.ave_options)
    lt.addWidget(self.auto_options)
    self.tool_options.setLayout(lt)

  def do_rdf_options(self):
    self.rdf_tool_options = QGroupBox('Options')
    lt = QHBoxLayout()

    self.rdf_ft_options = QGroupBox("Fourier Transform")
    layout = QFormLayout()
    self.rdf_ftsize = QLineEdit()
    self.rdf_ftsize.setValidator(QIntValidator())
    self.rdf_ftsize.setText(str(max(self.nprdf, 500)))
    layout.addRow(QLabel('FT bin size: '),self.rdf_ftsize)
    self.rdf_ft_options.setLayout(layout)

    self.toggle_rdf_ana_options(0)
    lt.addWidget(self.rdf_ft_options)
    self.rdf_tool_options.setLayout(lt)

  def do_msd_options(self):
    self.msd_tool_options = QGroupBox('Options')
    lt = QHBoxLayout()

    self.msd_grad_options = QGroupBox("Gradient")
    layout = QFormLayout()
    self.msd_gradsize = QLineEdit()
    self.msd_gradsize.setValidator(QIntValidator())
    self.msd_gradsize.setText(str(min(self.npmsd, 10)))
    layout.addRow(QLabel('Gradient block-averaging bin size: '),self.msd_gradsize)
    self.msd_grad_options.setLayout(layout)

    self.msd_hist_options = QGroupBox("Histogram")
    layout = QFormLayout()
    self.msdhist = QCheckBox()
    layout.addRow(QLabel('Plot histogram '),self.msdhist)
    self.msdbins = QLineEdit()
    self.msdbins.setValidator(QIntValidator())
    self.msdbins.setText('42')
    layout.addRow(QLabel('Histogram bins '),self.msdbins)
    self.msdpdf = QCheckBox()
    layout.addRow(QLabel('Normalised? '),self.msdpdf)
    self.msd_hist_options.setLayout(layout)

    self.toggle_msd_ana_options(0)
    lt.addWidget(self.msd_grad_options)
    lt.addWidget(self.msd_hist_options)
    self.msd_tool_options.setLayout(lt)


  def do_local_options(self):
    self.local_tool_options = QGroupBox('Options')
    lt = QHBoxLayout()

    self.local_plot_options = QGroupBox("Plotting")
    layout = QFormLayout()
    self.locallineplot = QCheckBox()
    layout.addRow(QLabel('Plot with line '),self.locallineplot)
    self.local_plot_options.setLayout(layout)

    self.local_ave_options = QGroupBox("Average")
    layout = QFormLayout()
    self.localave =QLabel()
    layout.addRow(QLabel('Average:  '),self.localave)
    self.local_ave_options.setLayout(layout)

    self.local_regress_options = QGroupBox("Regression")
    layout = QFormLayout()
    self.localregorder = QComboBox()
    self.localregorder.addItems(['Linear','Quadratic','Cubic'])
    self.localregorder.setCurrentIndex(0)
    layout.addRow(QLabel('Order:  '),self.localregorder)
    self.localregeqn = QCheckBox()
    layout.addRow(QLabel('Display equation:  '),self.localregeqn)
    self.local_regress_options.setLayout(layout)

    lt.addWidget(self.local_plot_options)
    lt.addWidget(self.local_ave_options)
    lt.addWidget(self.local_regress_options)
    self.local_tool_options.setLayout(lt)

  def do_local_ave(self,c,line,ix,iy,iz):
    self.axl.clear()
    if line==0:
        x = self.local_data[line][:,iy,iz]
        y = self.local_data[c+3][:,iy,iz]
        self.axl.set_xlabel('Position, '+r'$x$')
        self.axl.set_title(self.local_labels[c]+': y = {0:f}, z = {1:f}'.format(self.local_data[1][0,iy,iz], self.local_data[2][0,iy,iz]))
    elif line==1:
        x = self.local_data[line][ix,:,iz]
        y = self.local_data[c+3][ix,:,iz]
        self.axl.set_xlabel('Position, '+r'$y$')
        self.axl.set_title(self.local_labels[c]+': x = {0:f}, z = {1:f}'.format(self.local_data[0][ix,0,iz], self.local_data[2][ix,0,iz]))
    elif line==2:
        x = self.local_data[line][ix,iy,:]
        y = self.local_data[c+3][ix,iy,:]
        self.axl.set_xlabel('Position, '+r'$z$')
        self.axl.set_title(self.local_labels[c]+': x = {0:f}, y = {1:f}'.format(self.local_data[0][ix,iy,0], self.local_data[1][ix,iy,0]))

    mx = self.local_data[line][0,0,0] + self.local_data[line][self.nxlocal-1,self.nylocal-1,self.nzlocal-1]
    s = np.mean(y)
    if self.locallineplot.isChecked():
        self.axl.plot(x,y,'r-')
    else:
        self.axl.plot(x,y, 'r.')
    self.axl.axhline(y=s,color='blue',linestyle='--')

    self.axl.set_ylabel(self.local_datumAxes[c])
    self.axl.set_xlim(left=0.0,right=mx)
    self.local_plot.draw()

  def do_local_regress(self,c,line,ix,iy,iz):
    self.axl.clear()
    if line==0:
        x = self.local_data[line][:,iy,iz]
        y = self.local_data[c+3][:,iy,iz]
        self.axl.set_xlabel('Position, '+r'$x$')
        self.axl.set_title(self.local_labels[c]+': y = {0:f}, z = {1:f}'.format(self.local_data[1][0,iy,iz], self.local_data[2][0,iy,iz]))
    elif line==1:
        x = self.local_data[line][ix,:,iz]
        y = self.local_data[c+3][ix,:,iz]
        self.axl.set_xlabel('Position, '+r'$y$')
        self.axl.set_title(self.local_labels[c]+': x = {0:f}, z = {1:f}'.format(self.local_data[0][ix,0,iz], self.local_data[2][ix,0,iz]))
    elif line==2:
        x = self.local_data[line][ix,iy,:]
        y = self.local_data[c+3][ix,iy,:]
        self.axl.set_xlabel('Position, '+r'$z$')
        self.axl.set_title(self.local_labels[c]+': x = {0:f}, y = {1:f}'.format(self.local_data[0][ix,iy,0], self.local_data[1][ix,iy,0]))

    mx = self.local_data[line][0,0,0] + self.local_data[line][self.nxlocal-1,self.nylocal-1,self.nzlocal-1]
    self.axl.set_ylabel(self.local_datumAxes[c])
    self.axl.set_xlim(left=0.0,right=mx)

    # apply polynomial regression on data and add to plot

    depend = ['x','y','z']
    if len(y)>1:
        poly = Polynomial.fit(x,y,self.localregorder.currentIndex()+1).convert()
        formula = '$f ('+depend[line]+') = {0:f}'.format(poly.coef[0])
        if poly.coef[1]>=0.0:
            formula += ' + {0:f} '.format(poly.coef[1]) + depend[line]
        else:
            formula += ' - {0:f} '.format(-poly.coef[1]) + depend[line]
    
        if self.localregorder.currentIndex()>0:
            if poly.coef[2]>=0.0:
                formula += ' + {0:f} '.format(poly.coef[2]) + depend[line] +'^2'
            else:
                formula += ' - {0:f} '.format(-poly.coef[2]) + depend[line] +'^2'
    
        if self.localregorder.currentIndex()>1:
            if poly.coef[3]>=0.0:
                formula += ' + {0:f} '.format(poly.coef[3]) + depend[line] +'^3'
            else:
                formula += ' - {0:f} '.format(-poly.coef[3]) + depend[line] +'^3'
        formula +='$'
        fx = Polynomial(poly.coef)
    else:
        formula = '$f ('+depend[line]+')= {0:f}$'.format(y[0])
        fx = Polynomial(y[0])
    
    xx = np.linspace(0.0,mx)
    yy = fx(xx)

    # find regression coefficient based on fitted curve

    if len(y)>1:
        yhat = fx(x)
        ybar = np.sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y-ybar)**2)
        rsq = ssreg/sstot
    else:
        rsq = 1

    formula += "\n$R^2 = {0:f}$".format(rsq)

    self.axl.plot(x,y,'r.',xx,yy,'b--')
    if self.localregeqn.isChecked():
        self.axl.annotate(formula, xy=(0.05, 0.9), xycoords='axes fraction')
    
    self.local_plot.draw()


  def do_error(self,c):
      self.nwin.setEnabled(False)
      if c>0:
          self.nwin.setEnabled(True)
      self.do_run_ave()

  def toggle_ana_options(self,c):
    self.auto_options.setEnabled(False)
    self.ave_options.setEnabled(False)
    self.hist_options.setEnabled(False)
    if c == 1:
        self.hist_options.setEnabled(True)
    if c == 2:
        self.ave_options.setEnabled(True)
    if c == 3:
        self.auto_options.setEnabled(True)

  def toggle_rdf_ana_options(self,c):
    self.rdf_ft_options.setEnabled(False)
    if c == 1:
        self.rdf_ft_options.setEnabled(True)

  def toggle_msd_ana_options(self,c):
    self.msd_grad_options.setEnabled(False)
    self.msd_hist_options.setEnabled(False)
    if c == 1:
        self.msd_grad_options.setEnabled(True)
        self.msd_hist_options.setEnabled(True)


def readRDF(filename):
    try:
      title, header, rdfall = open(filename).read().split('\n',2)
    except IOError:
        return 0,0,0,[]
    ndata,npoints=map(int, header.split())
    b=3*npoints+2
    s=rdfall.split()
    nrdf=len(s)//b
    d=np.zeros((nrdf,npoints,3),dtype=float)
    labels=[]
    for i in range(nrdf):
      x=s[b*i:b*(i+1)]
      y=np.array(x[2:],dtype=float)
      y.shape= npoints,3
      d[i,:,:]=y
      if(x[0]=='all'):
        labels.append("all")
      else:
        labels.append(x[0]+" ... "+x[1])
    return nrdf+1,npoints,d,labels

def readMSD(filename):
    try:
      title, header, msdall = open(filename).read().split('\n',2)
    except IOError:
        return 0,0,0,[]
    npoints,ndata = map(int, header.split())
    b=3*npoints+1
    s=msdall.split()
    nmsd=ndata+1
    d=np.zeros((nmsd,npoints,3),dtype=float)
    labels=[]
    for i in range(nmsd):
      if(s[b*i]=='all'):
        labels.append("all species")
        x=s[b*i+2:]
      else:
        labels.append(s[b*i])
        x=s[b*i+1:b*(i+1)]
      y=np.array(x,dtype=float)
      y.shape= npoints,3
      d[i,:,:]=y
    return nmsd+1,npoints,d,labels

def readLocal(filename):
    if not os.path.isfile(filename+".vts") and not os.path.isfile(filename+".vtk"):
       return 0,0,0,0,0,[],[],[]
    
    xmlvtk = os.path.isfile(filename+".vts")
    try:
      if xmlvtk:
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(filename+".vts")
      else:
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(filename+".vtk")
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
      reader.Update()
    except IOError:
       return 0,0,0,0,0,[],[],[]
    data = reader.GetOutput()
    # find spacing between sections and numbers
    n = data.GetExtent()
    numX = n[1] - n[0]
    numY = n[3] - n[2]
    numZ = n[5] - n[4]
    if xmlvtk:
      s = data.GetBounds()
      dx = (s[1] - s[0]) / float(numX)
      dy = (s[3] - s[2]) / float(numY)
      dz = (s[5] - s[4]) / float(numZ)
    else:
      s = data.GetSpacing()
      dx = s[0]
      dy = s[1]
      dz = s[2]

    datumNames=[]
    datumShortNames=[]
    datumAxes=[]
    numdata = data.GetCellData().GetNumberOfArrays()
    d = []

    # add grid cell centres as first three datasets

    x = np.zeros((numX,numY,numZ),dtype=float)
    y = np.zeros((numX,numY,numZ),dtype=float)
    z = np.zeros((numX,numY,numZ),dtype=float)
    for kk in range(numZ):
       zz = (float(kk) + 0.5) * dz
       for jj in range(numY):
          yy = (float(jj) + 0.5) * dy
          for ii in range(numX):
             xx = (float(ii) + 0.5) * dx
             x[ii,jj,kk] = xx
             y[ii,jj,kk] = yy
             z[ii,jj,kk] = zz
 
    d.append(x)
    d.append(y)
    d.append(z)

    # add all other available datasets (splitting velocities
    # into separate Cartesian components)

    for i in range(numdata):
        namedata = data.GetCellData().GetArrayName(i)
        dataset = VN.vtk_to_numpy(data.GetCellData().GetArray(i))
        if namedata == 'velocity':
            dataset = np.hsplit(dataset, 3)
            vx = dataset[0].reshape(numZ, numY, numX).transpose(2, 1, 0)
            vy = dataset[1].reshape(numZ, numY, numX).transpose(2, 1, 0)
            vz = dataset[2].reshape(numZ, numY, numX).transpose(2, 1, 0)
            d.append(vx)
            d.append(vy)
            d.append(vz)
            datumNames.append("Velocity (x-component)")
            datumNames.append("Velocity (y-component)")
            datumNames.append("Velocity (z-component)")
            datumShortNames.append('velocity_x')
            datumShortNames.append('velocity_y')
            datumShortNames.append('velocity_z')
            datumAxes.append('Velocity (x-component), $v_x$')
            datumAxes.append('Velocity (y-component), $v_y$')
            datumAxes.append('Velocity (z-component), $v_z$')
        elif namedata.startswith('density'):
            d.append(dataset.reshape(numZ,numY,numX).transpose(2, 1, 0))
            comp = namedata[8:]
            datumNames.append("Particle density ("+comp+")")
            datumShortNames.append(namedata)
            datumAxes.append('Particle density, '+r'$\rho $')
        elif namedata.startswith('bead_numbers'):
            d.append(dataset.reshape(numZ,numY,numX).transpose(2, 1, 0))
            datumNames.append("Number of particles")
            datumShortNames.append(namedata)
            datumAxes.append('Number of particles, $N$')
        elif namedata == 'temperature':
            d.append(dataset.reshape(numZ,numY,numX).transpose(2, 1, 0))
            datumNames.append("Temperature")
            datumShortNames.append(namedata)
            datumAxes.append('Temperature, $T$')
        elif namedata.startswith('temperature_'):
            d.append(dataset.reshape(numZ,numY,numX).transpose(2, 1, 0))
            comp = namedata[-1:]
            datumNames.append("Partial temperature ("+comp+"-component)")
            datumShortNames.append(namedata)
            datumAxes.append('Partial temperature ('+comp+'-component), $T_'+comp+'$')
        elif namedata.startswith('pressure'):
            d.append(dataset.reshape(numZ,numY,numX).transpose(2, 1, 0))
            comp = namedata[-2:]
            datumNames.append("Pressure tensor ("+comp+"-component)")
            datumShortNames.append(namedata)
            datumAxes.append('Pressure tensor ('+comp+'-component), $P_{'+comp+'}')
        elif namedata.startswith('species'):
            d.append(dataset.reshape(numZ,numY,numX).transpose(2, 1, 0))
            comp = namedata[8:]
            datumNames.append("Volume fraction of species "+comp)
            datumShortNames.append(namedata)
            datumAxes.append('Volume fraction, '+r'$\phi$')
        elif namedata.startswith('molecule'):
            d.append(dataset.reshape(numZ,numY,numX).transpose(2, 1, 0))
            comp = namedata[9:]
            datumNames.append("Volume fraction of molecule "+comp)
            datumShortNames.append(namedata)
            datumAxes.append('Volume fraction, '+r'$\phi$')

    return numX,numY,numZ,len(datumNames),d,datumNames,datumShortNames,datumAxes


def readCorrel(filename):
  try:
    h, s = open(filename).read().split('\n',1)
  except IOError:
      return 0,0,0,[],[] 
  names = h.split()
  nd = len(names) - 1
  if(names[0]=='#'):
    nd = nd - 1
  datumNames=[]
  datumAxes=[]
  for i in range(len(names)):
    if(names[i]=='en-total'):
        datumNames.append("total system energy per particle")
        datumAxes.append('$E_{tot}$ [$k_B T$]')
    elif(names[i]=='pe-total'):
        datumNames.append("total potential energy per particle")
        datumAxes.append('$E_{pot}$ [$k_B T$]')
    elif(names[i]=='ee-total'):
        datumNames.append("electrostatic energy per particle")
        datumAxes.append('$E_{elec}$ [$k_B T$]')
    elif(names[i]=='se-total'):
        datumNames.append("surface energy per particle")
        datumAxes.append('$E_{surf}$ [$k_B T$]')
    elif(names[i]=='be-total'):
        datumNames.append("bond energy per particle")
        datumAxes.append('$E_{bond}$ [$k_B T$]')
    elif(names[i]=='ae-total'):
        datumNames.append("angle energy per particle")
        datumAxes.append('$E_{ang}$ [$k_B T$]')
    elif(names[i]=='de-total'):
        datumNames.append("dihedral energy per particle")
        datumAxes.append('$E_{dihed}$ [$k_B T$]')
    elif(names[i]=='pressure'):
        datumNames.append("pressure")
        datumAxes.append('$P$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_xx' or names[i]=='p_xx'):
        datumNames.append("pressure tensor xx-component")
        datumAxes.append('$P_{xx}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_xy' or names[i]=='p_xy'):
        datumNames.append("pressure tensor xy-component")
        datumAxes.append('$P_{xy}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_xz' or names[i]=='p_xz'):
        datumNames.append("pressure tensor xz-component")
        datumAxes.append('$P_{xz}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_yx' or names[i]=='p_yx'):
        datumNames.append("pressure tensor yx-component")
        datumAxes.append('$P_{yx}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_yy' or names[i]=='p_yy'):
        datumNames.append("pressure tensor yy-component")
        datumAxes.append('$P_{yy}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_yz' or names[i]=='p_yz'):
        datumNames.append("pressure tensor yz-component")
        datumAxes.append('$P_{yz}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_zx' or names[i]=='p_zx'):
        datumNames.append("pressure tensor zx-component")
        datumAxes.append('$P_{zx}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_zy' or names[i]=='p_zy'):
        datumNames.append("pressure tensor zy-component")
        datumAxes.append('$P_{zy}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='s_zz' or names[i]=='p_zz'):
        datumNames.append("pressure tensor zz-component")
        datumAxes.append('$P_{zz}$ [$k_B T \ell_0^{-3}$]')
    elif(names[i]=='volume'):
        datumNames.append("volume")
        datumAxes.append('$V$ [$\ell_0^3$]')
    elif(names[i]=='L_x'):
        datumNames.append("box size x-component")
        datumAxes.append('$L_x$ [$\ell_0$]')
    elif(names[i]=='L_y'):
        datumNames.append("box size y-component")
        datumAxes.append('$L_y$ [$\ell_0$]')
    elif(names[i]=='L_z'):
        datumNames.append("box size z-component")
        datumAxes.append('$L_z$ [$\ell_0$]')
    elif(names[i]=='tension'):
        datumNames.append("z-component interfacial tension")
        datumAxes.append(r'$\gamma_z$'+' [$k_B T \ell_0^{-2}$]')
    elif(names[i]=='temperature'):
        datumNames.append("system temperature")
        datumAxes.append('$T$ [$k_B T$]')
    elif(names[i]=='temp-x'):
        datumNames.append("partial temperature x-component")
        datumAxes.append('$T_x$ [$k_B T$]')
    elif(names[i]=='temp-y'):
        datumNames.append("partial temperature y-component")
        datumAxes.append('$T_y$ [$k_B T$]')
    elif(names[i]=='temp-z'):
        datumNames.append("partial temperature z-component")
        datumAxes.append('$T_z$ [$k_B T$]')
    elif(names[i]=='bndlen-av'):
        datumNames.append("mean bond length")
        datumAxes.append(r'$\langle r_{ij} \rangle$'+' [$\ell_0$]')
    elif(names[i]=='bndlen-max'):
        datumNames.append("maximum bond length")
        datumAxes.append('$r_{ij,max}$ [$\ell_0$]')
    elif(names[i]=='bndlen-min'):
        datumNames.append("minimum bond length")
        datumAxes.append('$r_{ij,min}$ [$\ell_0$]')
    elif(names[i]=='angle-av'):
        datumNames.append("mean bond angle")
        datumAxes.append(r'$\langle \theta_{ijk} \rangle$'+' [°]')
    elif(names[i]=='dihed-av'):
        datumNames.append("mean bond dihedral")
        datumAxes.append(r'$\langle \phi_{ijkm} \rangle$'+' [°]')
  d = np.array(s.split(), dtype=float)
  n = d.size//(nd+1)
  d.shape = n, nd+1
  return n,nd,d,datumNames,datumAxes

if __name__ == '__main__':
  app = QApplication(sys.argv)
  exe = App()
  sys.exit(app.exec_())
