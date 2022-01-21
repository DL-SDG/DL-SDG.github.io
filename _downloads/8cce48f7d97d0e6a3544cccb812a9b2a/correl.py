#!/usr/bin/env python3 
# -*- coding: UTF-8 -*-

__author__ = "Alin Marin Elena <alin@elena.space> and Michael Seaton <michael.seaton@stfc.ac.uk>"
__copyright__ = "Copyright© 2019 Alin M Elena and Michael Seaton"
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
    

class App(QMainWindow):
  def __init__(self):
    super().__init__()
    self.title="DL_MESO_DPD CORREL file explorer"
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
    n,nd,data,names = readCorrel(filename="CORREL")
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
    #self.tab_advanced = QWidget()

    self.tabs.addTab(self.tab_correl,"CORREL")

    self.tab_correl.layout = QVBoxLayout(self)
    self.plot = FigureCanvas(Figure(figsize=(16,12)))
    self.tab_correl.layout.addWidget(NavigationToolbar(self.plot, self))
    self.tab_correl.layout.addWidget(self.plot)
    self.tab_correl.layout.addStretch(1)
    self.n,self.nd,self.data,self.datumNames = readCorrel()
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
    self.axr.set_xlabel('Reciprocal distance [DPD length units]^(-1)')
    self.axr.set_ylabel('FT of g(r)')
    self.axr.set_title("Structure factor of "+self.rdf_labels[c])
    self.axr.plot(freq,2*np.abs(ft)/f,'g-')
    self.rdf_plot.draw()

   
  def update_plot(self,c):
    self.ax.clear()
    self.ax.plot(self.data[:,0],self.data[:,c+1], 'r-')
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel('Time [DPD time units]')
    self.ax.set_ylabel('check units')
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


    self.ave.setText("{0:.8E} +/- {1:.8E}".format(v[k-1],se))  
    self.ax.plot(self.data[:,0],self.data[:,c+1], 'r-',x,v,'b-')
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel('Time [DPD time units]')
    self.ax.set_ylabel('check units')
    self.ax.set_xlim(left=min(self.data[:,0]),right=max(self.data[:,0]))
    self.plot.draw()


  def do_autocorr(self,l=-1):
    self.ax.clear()
    l = int(self.lag.text())
    if l<1 or l>self.n:
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle('WARNING')
        msgbox.setText('Lag out of range: must be between 1 and {0:d}'.format(self.n))
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()
        return
    c= self.tsb.currentIndex()
    av = np.average(self.data[:,c+1])
    var = np.var(self.data[:,c+1])
    y=self.data[:,c+1]-av
    n=len(y)
    x=np.arange(0,l,1)
    ci=np.correlate(y,y,mode='full')[-n:]
    ci=np.array([(y[:n-k]*y[-(n-k):]).sum() for k in range(l)])
    ci=ci/(var*(np.arange(n,n-l,-1)-x))
    
    self.ax.plot(x,ci, 'r-')
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel('Lag')
    self.ax.set_ylabel('')
    self.ax.set_xlim(left=0,right=l)
    self.plot.draw()

  def do_norm(self,c):
      self.do_histogram()
  
  def do_histogram(self,c=-1):
    if c == -1 :
        c= self.tsb.currentIndex()
    self.ax.clear()
    hist,bins = np.histogram(self.data[:,c+1],bins=int(self.bins.text()),density=self.pdf.isChecked())
    self.ax.set_title(self.datumNames[c])
    self.ax.set_xlabel('bins')
    self.ax.set_ylabel('pdf')
    self.ax.bar(bins[:-1],hist)
    self.plot.draw()

  def do_ft(self,c):
    self.ax.clear()
    ft = np.fft.rfft(self.data[:,c+1])
    d=self.data[1,0]-self.data[0,0]
    freq = np.fft.rfftfreq(self.n, d=d)
    self.ax.set_xlabel('Frequency [DPD time units]^(-1)')
    self.ax.set_ylabel('')
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

def readCorrel(filename="CORREL"):
  h, s = open(filename).read().split('\n',1)
  names = h.split()
  nd = len(names) - 1
  if(names[0]=='#'):
    nd = nd - 1
  datumNames=[]
  for i in range(len(names)):
    if(names[i]=='en-total'):
        datumNames.append("total system energy")
    elif(names[i]=='pe-total'):
        datumNames.append("total potential energy")
    elif(names[i]=='ee-total'):
        datumNames.append("electrostatic energy")
    elif(names[i]=='se-total'):
        datumNames.append("surface energy")
    elif(names[i]=='be-total'):
        datumNames.append("bond energy")
    elif(names[i]=='ae-total'):
        datumNames.append("angle energy")
    elif(names[i]=='de-total'):
        datumNames.append("dihedral energy")
    elif(names[i]=='pressure'):
        datumNames.append("pressure")
    elif(names[i]=='s_xx' or names[i]=='p_xx'):
        datumNames.append("pressure tensor xx-component")
    elif(names[i]=='s_xy' or names[i]=='p_xy'):
        datumNames.append("pressure tensor xy-component")
    elif(names[i]=='s_xz' or names[i]=='p_xz'):
        datumNames.append("pressure tensor xz-component")
    elif(names[i]=='s_yx' or names[i]=='p_yx'):
        datumNames.append("pressure tensor yx-component")
    elif(names[i]=='s_yy' or names[i]=='p_yy'):
        datumNames.append("pressure tensor yy-component")
    elif(names[i]=='s_yz' or names[i]=='p_yz'):
        datumNames.append("pressure tensor yz-component")
    elif(names[i]=='s_zx' or names[i]=='p_zx'):
        datumNames.append("pressure tensor zx-component")
    elif(names[i]=='s_zy' or names[i]=='p_zy'):
        datumNames.append("pressure tensor zy-component")
    elif(names[i]=='s_zz' or names[i]=='p_zz'):
        datumNames.append("pressure tensor zz-component")
    elif(names[i]=='volume'):
        datumNames.append("volume")
    elif(names[i]=='L_x'):
        datumNames.append("box size x-component")
    elif(names[i]=='L_y'):
        datumNames.append("box size y-component")
    elif(names[i]=='L_z'):
        datumNames.append("box size z-component")
    elif(names[i]=='tension'):
        datumNames.append("z-component interfacial tension")
    elif(names[i]=='temperature'):
        datumNames.append("system temperature")
    elif(names[i]=='temp-x'):
        datumNames.append("partial temperature x-component")
    elif(names[i]=='temp-y'):
        datumNames.append("partial temperature y-component")
    elif(names[i]=='temp-z'):
        datumNames.append("partial temperature z-component")
    elif(names[i]=='bndlen-av'):
        datumNames.append("mean bond length")
    elif(names[i]=='bndlen-max'):
        datumNames.append("maximum bond length")
    elif(names[i]=='bndlen-min'):
        datumNames.append("minimum bond length")
    elif(names[i]=='angle-av'):
        datumNames.append("mean bond angle")
    elif(names[i]=='dihed-av'):
        datumNames.append("mean bond dihedral")
  d = np.array(s.split(), dtype=float)
  n = d.size//(nd+1)
  d.shape = n, nd+1
  return n,nd,d,datumNames

if __name__ == '__main__':
  app = QApplication(sys.argv)
  exe = App()
  sys.exit(app.exec_())
