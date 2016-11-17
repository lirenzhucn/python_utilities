#!/usr/bin/env python

from PyQt4.QtGui import (QLabel, QDialog, QPixmap, QPainter, QColor,
                         QScrollBar, QPushButton, QVBoxLayout, QHBoxLayout,
                         QWidget, QImage, QMenu)
from PyQt4.QtCore import Qt, pyqtSlot, SIGNAL
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import register_cmap

from ._custom_colormaps import _kwave_data, _green_data


# some constants
kwave_cm = ListedColormap(_kwave_data, name='kwave')
green_cm = LinearSegmentedColormap('green', _green_data)
register_cmap(name='kwave', cmap=kwave_cm)
register_cmap(name='green', cmap=green_cm)


class DoubleClickableLabel(QLabel):
    """A QLabel that sends out doubleClicked signal"""
    __pyqtSignals__ = ('doubleClicked()')

    def mouseDoubleClickEvent(self, event):
        self.emit(SIGNAL('doubleClicked()'))


class TGCDialog(QDialog):
    """A dialog for manually tuning time gain compensation
    """

    __pyqtSignals__ = ('TGCChanged()')

    def __init__(self, numFrames, parent=None,
                 maxVal=10.0, sliderDensity=10):
        QDialog.__init__(self, parent)
        self.numFrames = numFrames
        self.maxVal = maxVal
        self.sliderDensity = sliderDensity
        self.setupUi()
        # signals and slots
        self.mBtnClose.clicked.connect(self.accept)

    def _createSliderAndLabel(self, i):
            s = QScrollBar(Qt.Horizontal, self)
            s.setPageStep(1)
            s.setSingleStep(1)
            s.setMinimum(0)
            s.setMaximum(100)
            s.setValue(0)
            l = QLabel('frame %3d' % i, self)
            l.setAlignment(Qt.AlignLeft)
            return s, l

    def setupUi(self):
        # construct a slider every self.sliderDensity frames
        self.labels = []
        self.sliders = []
        self.nodes = list(range(0, self.numFrames, self.sliderDensity))
        # make sure a node is created for the end point
        if self.numFrames - 1 not in self.nodes:
            self.nodes.append(self.numFrames - 1)
        for i in self.nodes:
            s, l = self._createSliderAndLabel(i)
            self.sliders.append(s)
            self.labels.append(l)
        # button(s)
        self.mBtnClose = QPushButton('Close', self)
        # layout
        vlayout = QVBoxLayout(self)
        for l, s in zip(self.labels, self.sliders):
            hlayout = QHBoxLayout()
            hlayout.addWidget(l)
            hlayout.setStretchFactor(l, 1)
            hlayout.addWidget(s)
            hlayout.setStretchFactor(s, 5)
            vlayout.addLayout(hlayout)
        vlayout.addWidget(self.mBtnClose)

    @property
    def TGCVector(self):
        """get the TGC vector that consists of one entry for each frame"""
        t = np.array(self.nodes)
        g = np.array([s.value() for s in self.sliders], dtype=np.float64)
        g = np.power(self.maxVal, g / 100.0)
        ipl = interp1d(t, g)
        return ipl(np.array(range(0, self.numFrames)))


class MinMaxDialog(QDialog):

    __pyqtSignals__ = ('minMaxChanged()')

    def __init__(self, dMin, dMax, imgStat, parent=None):
        QDialog.__init__(self, parent)
        self.dMin = dMin
        self.dMax = dMax
        self.imgStat = imgStat
        self.setupUi()
        # signals and slots
        self.mBtnClose.clicked.connect(self.accept)
        self.mScMin.valueChanged.connect(self.minMaxChange)
        self.mScMax.valueChanged.connect(self.minMaxChange)

    def setupHistogram(self):
        hist = list(self.imgStat.hist)
        hist = [float(v)/max(hist) for v in hist]
        width = len(hist)
        height = len(hist)/2
        self.mLbHist = QLabel(self)
        self.mLbHist.setFixedSize(width, height)
        self.mPixmapHist = QPixmap(width, height)
        self.mPixmapHist.fill()
        qp = QPainter()
        qp.begin(self.mPixmapHist)
        qp.setPen(QColor(100, 100, 100))
        for ind in range(len(hist)):
            qp.drawLine(ind, height, ind, (1-hist[ind])*height)
        qp.end()
        # self.mLbHist.setPixmap(self.mPixmapHist)
        self.drawHistLabel()

    def drawHistLabel(self):
        width = self.mPixmapHist.width()
        height = self.mPixmapHist.height()
        lp = int((self.dMin-self.imgStat.min)/self.imgStat.range*width)
        rp = int((self.dMax-self.imgStat.min)/self.imgStat.range*width)
        pixmap = QPixmap(width, height)
        qp = QPainter()
        qp.begin(pixmap)
        qp.drawPixmap(0, 0, self.mPixmapHist)
        qp.setPen(QColor(0, 0, 0))
        qp.drawLine(lp, height, rp, 0)
        qp.end()
        self.mLbHist.setPixmap(pixmap)

    def setupUi(self):
        self.setWindowTitle('Min & Max')
        # histogram
        self.setupHistogram()
        # sliders
        self.mScMin = QScrollBar(Qt.Horizontal, self)
        self.mScMin.setPageStep(1)
        self.mScMin.setSingleStep(1)
        self.mScMin.setMinimum(0)
        self.mScMin.setMaximum(100)
        minVal = int((self.dMin-self.imgStat.min)/self.imgStat.range*100)
        self.mScMin.setValue(minVal)
        lbMin = QLabel('Minimum', self)
        lbMin.setAlignment(Qt.AlignCenter)
        self.mScMax = QScrollBar(Qt.Horizontal, self)
        self.mScMax.setPageStep(1)
        self.mScMax.setSingleStep(1)
        self.mScMax.setMinimum(0)
        self.mScMax.setMaximum(100)
        maxVal = int((self.dMax-self.imgStat.min)/self.imgStat.range*100)
        self.mScMax.setValue(maxVal)
        lbMax = QLabel('Maximum', self)
        lbMax.setAlignment(Qt.AlignCenter)
        # buttons
        self.mBtnClose = QPushButton('Close', self)
        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.mLbHist)
        layout.addWidget(self.mScMin)
        layout.addWidget(lbMin)
        layout.addWidget(self.mScMax)
        layout.addWidget(lbMax)
        layout.addWidget(self.mBtnClose)

    @pyqtSlot(int)
    def minMaxChange(self, int):
        self.dMin = self.mScMin.value()/100.0*self.imgStat.range +\
            self.imgStat.min
        self.dMax = self.mScMax.value()/100.0*self.imgStat.range +\
            self.imgStat.min
        self.drawHistLabel()
        self.mLbHist.update()
        self.emit(SIGNAL('minMaxChanged()'))

    @property
    def results(self):
        return (float(self.dMin), float(self.dMax))


class ImageStat:
    '''Just like PIL's ImageStat.Stat class, this class provides lazily
    evaluated attributes about image statistics
    '''

    HIST_BIN_NUM = 128

    def __init__(self, imgData):
        self.imgData = imgData
        self._min = None
        self._max = None
        self._numSlices = None
        self._width = None
        self._height = None
        self._hist = None

    @property
    def extrema(self):
        return (self.min, self.max)

    @property
    def min(self):
        if self._min is None:
            self._min = np.amin(self.imgData)
        return self._min

    @property
    def max(self):
        if self._max is None:
            self._max = np.amax(self.imgData)
        return self._max

    @property
    def numSlices(self):
        if self._numSlices is None:
            self._numSlices = self.imgData.shape[2]
        return self._numSlices

    @property
    def imgSize(self):
        return (self.width, self.height)

    @property
    def width(self):
        if self._width is None:
            self._width = self.imgData.shape[1]
        return self._width

    @property
    def height(self):
        if self._height is None:
            self._height = self.imgData.shape[0]
        return self._height

    @property
    def hist(self):
        if self._hist is None:
            self._hist, junk = np.histogram(self.imgData, self.HIST_BIN_NUM)
        return self._hist

    @property
    def range(self):
        return self.max - self.min


class ROI():
    '''The class for storing and drawing ROI'''

    def __init__(self, roi):
        assert(len(roi) == 4)
        self.roi = roi

    def draw(self, pixmap):
        # pixmap should be QPixmap
        pass


class ImageSliceDisplay(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.dMin = 0.0
        self.dMax = 0.0
        self.imgData = None
        self.imgStat = None
        # the rectangle representing ROI
        self.roi = ROI((0, 0, 0, 0))
        self.tgcDialog = None
        self.mmDialog = None
        self.setupUi()

    def updateStatus(self):
        msg = '%d/%d; %d x %d; min: %.6f, max: %.6f' %\
         (self.mScSlice.value()+1, self.imgStat.numSlices,
          self.imgStat.width, self.imgStat.height,
          self.dMin, self.dMax)
        self.mLbStatus.setText(msg)

    def setupUi(self):
        self.mLbStatus = QLabel(self)
        self.mLbDisplay = DoubleClickableLabel(self)
        self.mScSlice = QScrollBar(Qt.Horizontal, self)
        self.mScSlice.setPageStep(1)
        self.mScSlice.setMinimum(0)
        self.mScSlice.setMaximum(0)
        self.mScSlice.setSingleStep(1)
        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.mLbStatus)
        layout.addWidget(self.mLbDisplay)
        layout.addWidget(self.mScSlice)
        # signal/slot pairs
        self.mScSlice.valueChanged.connect(self.onSliceChanged)
        self.connect(self.mLbDisplay, SIGNAL('doubleClicked()'),
                     self.onDisplayDoubleClicked)
        self.mLbDisplay.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mLbDisplay.customContextMenuRequested.connect(
            self.openDisplayContextMenu)

    def openDisplayContextMenu(self, pos):
        menu = QMenu()
        minMaxAction = menu.addAction('Display range ...')
        tgcAction = menu.addAction('TGC ...')
        action = menu.exec_(self.mLbDisplay.mapToGlobal(pos))
        if action == minMaxAction:
            self.onDisplayDoubleClicked()
        elif action == tgcAction:
            self.onTGCEdit()

    def setRange(self, vmin, vmax):
        self.dMin, self.dMax = vmin, vmax

    @pyqtSlot(int)
    def onSliceChanged(self, val):
        self.updateStatus()
        self.prepareQImage(val)
        self.update()

    @pyqtSlot()
    def minMaxChange(self):
        self.dMin, self.dMax = self.mmDialog.results
        self.prepareQImage(self.mScSlice.value())
        self.updateStatus()
        self.update()

    @pyqtSlot()
    def onDisplayDoubleClicked(self):
        self.mmDialog = MinMaxDialog(self.dMin, self.dMax, self.imgStat, self)
        self.connect(self.mmDialog, SIGNAL('minMaxChanged()'),
                     self.minMaxChange)
        self.mmDialog.exec_()
        self.disconnect(self.mmDialog, SIGNAL('minMaxChanged()'),
                        self.minMaxChange)

    def onTGCEdit(self):
        im = self.orgImgData
        if self.tgcDialog is None:
            self.tgcDialog = TGCDialog(im.shape[2])
        self.tgcDialog.exec_()
        tgcVector = self.tgcDialog.TGCVector
        im = im * tgcVector[None, None, :]
        # we only update certain things
        self.imgData = im
        self.imgStat = ImageStat(self.imgData)
        self.dMin, self.dMax = self.imgStat.extrema
        self.updateStatus()
        self.prepareQImage(self.mScSlice.value())
        self.update()

    def setInput(self, imgData, cmapName):
        self.orgImgData = imgData
        self.imgData = imgData
        self.imgStat = ImageStat(imgData)
        self.dMin, self.dMax = self.imgStat.extrema
        self.cmapName = cmapName
        # setup scroll bar
        self.mScSlice.setMaximum(imgData.shape[2] - 1)
        self.mScSlice.setValue(0)
        # setup Label size
        self.mLbDisplay.setFixedSize(imgData.shape[1], imgData.shape[0])
        # setup display image
        self.updateStatus()
        self.prepareQImage(0)
        self.update()

    def prepareQImage(self, ind):
        img = self.imgData[:, :, ind]
        scaledImg = (img - self.dMin) / (self.dMax - self.dMin)
        scaledImg[scaledImg < 0.0] = 0.0
        scaledImg[scaledImg > 1.0] = 1.0
        cmap = plt.get_cmap(self.cmapName)
        rgbaImg_temp = cmap(scaledImg, bytes=True)
        rgbaImg = np.zeros(rgbaImg_temp.shape, dtype=np.uint8)
        rgbaImg[:, :, 0] = rgbaImg_temp[:, :, 2]
        rgbaImg[:, :, 1] = rgbaImg_temp[:, :, 1]
        rgbaImg[:, :, 2] = rgbaImg_temp[:, :, 0]
        rgbaImg[:, :, 3] = 255
        self.qimg = QImage(rgbaImg.tostring(order='C'),
                           rgbaImg.shape[1], rgbaImg.shape[0],
                           QImage.Format_RGB32)
        pix = QPixmap.fromImage(self.qimg)
        self.mLbDisplay.setPixmap(pix.scaled(self.mLbDisplay.size(),
                                             Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))
