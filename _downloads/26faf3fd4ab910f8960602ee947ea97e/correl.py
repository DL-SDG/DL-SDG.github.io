#!/usr/bin/env python3 
# -*- coding: UTF-8 -*-

__author__ = "Alin Marin Elena <alin@elena.space> and Michael Seaton <michael.seaton@stfc.ac.uk>"
__copyright__ = "Copyright© 2019, 2022 Alin M Elena and Michael Seaton"
__license__ = "GPL-3.0-only"
__version__ = "1.0"
__description__ = "Modification of statis.py at https://gitlab.com/drFaustroll/dlTables"

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
from scipy.stats import gaussian_kde
    

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
    exit = QAction(QIcon('exit.png'), 'Exit', self)
    exit.setShortcut('Ctrl+Q')
    exit.setStatusTip('Exit application')
    exit.triggered.connect(self.close)
    fileMenu.addAction(exit)

    exit = QAction(QIcon('save.png'), 'Flatten CORREL', self)
    exit.setShortcut('Ctrl+F')
    exit.setStatusTip('Flatten CORREL file in timeseries')
    exit.triggered.connect(self.flatten)
    fileMenu.addAction(exit)
    
    self.show()

  def flatten(self):
    n,nd,data,names,axes = readCorrel(filename="CORREL")
    for i in range(nd):
      with open(names[i],'w') as f:
        for j in range(n):
            f.write("{0} {1}\n".format(data[j,0],data[j,i+1]))

class MyTabs(QWidget):
  def __init__(self,parent):
    super(QWidget, self).__init__(parent)

    self.layout = QVBoxLayout()

    self.tabs = QTabWidget() 
    self.tab_correl = QWidget()
    self.tab_rdf = QWidget()
    self.tab_msd = QWidget()
    #self.tab_advanced = QWidget()

    self.tabs.addTab(self.tab_correl,"CORREL")

    self.tab_correl.layout = QVBoxLayout(self)
    self.plot = FigureCanvas(Figure(figsize=(16,12)))
    self.tab_correl.layout.addWidget(NavigationToolbar(self.plot, self))
    self.tab_correl.layout.addWidget(self.plot)
    self.tab_correl.layout.addStretch(1)
    self.n,self.nd,self.data,self.datumNames,self.datumAxes = readCorrel()
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

    self.tabs.addTab(self.tab_rdf,"RDFDAT")
    self.tab_rdf.layout = QVBoxLayout(self)
    self.rdf_plot = FigureCanvas(Figure(figsize=(16,12)))
    self.tab_rdf.layout.addWidget(NavigationToolbar(self.rdf_plot, self))
    self.tab_rdf.layout.addWidget(self.rdf_plot)
    self.tab_rdf.layout.addStretch(1) 
    self.nrdf,self.nprdf,self.rdf_data,self.rdf_labels=readRDF(filename="RDFDAT")
    self.axr = self.rdf_plot.figure.add_subplot(111)
    if self.nrdf>0 :
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

    self.tabs.addTab(self.tab_msd,"MSDDAT")
    self.tab_msd.layout = QVBoxLayout(self)
    self.msd_plot = FigureCanvas(Figure(figsize=(16,12)))
    self.tab_msd.layout.addWidget(NavigationToolbar(self.msd_plot, self))
    self.tab_msd.layout.addWidget(self.msd_plot)
    self.tab_msd.layout.addStretch(1)
    self.nmsd,self.npmsd,self.msd_data,self.msd_labels=readMSD(filename="MSDDAT")
    self.axm = self.msd_plot.figure.add_subplot(111)
    if self.nmsd>0 :
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

    self.layout.addWidget(self.tabs)
    self.setLayout(self.layout)

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
    ft = np.zeros(f,dtype=float)
    ft[0:self.nprdf] = self.rdf_data[c,0:self.nprdf,1]-1.0
    ft = np.fft.rfft(ft)
    d=self.rdf_data[c,1,0]-self.rdf_data[c,0,0]
    freq = np.fft.rfftfreq(f, d=d)
    self.axr.set_xlabel('Reciprocal distance, $r^{-1}$ [$\ell_0^{-1}$]')
    self.axr.set_ylabel('FT of g(r), '+r'$S \left(r^{-1}\right)$')
    self.axr.set_title("Structure factor of "+self.rdf_labels[c])
    self.axr.plot(freq,2*np.abs(ft)/f,'g-')
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


def readRDF(filename="RDFDAT"):
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

def readMSD(filename="MSDDAT"):
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

def readCorrel(filename="CORREL"):
  h, s = open(filename).read().split('\n',1)
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
