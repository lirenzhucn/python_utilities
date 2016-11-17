"""Provide a polymask function to display a 2D image and let the user to select
a polygon as the mask, then return the mask.
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import numpy as np
from matplotlib import cm


def _compute_init_box(im_shape):
    xsize, ysize = im_shape
    x0 = xsize // 4
    x1 = x0 + xsize // 2
    y0 = ysize // 4
    y1 = y0 + ysize // 2
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _convert_color(im, color_map):
    im = im.astype(np.float64)
    im = im / im.max()
    cmObj = cm.get_cmap(color_map)
    return cmObj(im)


def polymask(im, return_type='mask', win_title='polymask', color_map='gray',
             pen=(0, 9)):
    """The main interface function.
    im          - the input image, assumed 2D
    return_type - 'mask' means to return a binary image of the size of im
                  'boundary' means to return a tuple of vertices of the polygon
    win_title   - the string you want to appear as the window's title
    pen         - pen option passed to ROI object
    """
    im = im.T
    assert(len(im.shape) == 2)
    win_size = [e + 10 for e in im.shape]
    app = QtGui.QApplication([])
    widget = pg.GraphicsWindow(size=win_size, border=True)
    widget.setWindowTitle(win_title)
    widgetLayout = widget.addLayout(row=0, col=0)
    view = widgetLayout.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
    imItem = pg.ImageItem(_convert_color(im, color_map))
    view.addItem(imItem)
    view.disableAutoRange('xy')
    view.autoRange()
    roi = pg.PolyLineROI(_compute_init_box(im.shape), pen=pen, closed=True)
    view.addItem(roi)
    app.exec_()
    # compute boundary
    basePos = roi.pos()
    boundary = [(pos.x() + basePos[0], pos.y() + basePos[1]) for name, pos in
                roi.getLocalHandlePositions()]
    # compute mask
    mask = np.zeros(im.shape, dtype=np.bool)
    sliceObj, _ = roi.getArraySlice(mask, imItem)
    maskSlice = roi.getArrayRegion(np.logical_not(mask), imItem)
    mask[sliceObj] = maskSlice
    mask = mask.T
    if return_type == 'boundary':
        return boundary
    elif return_type == 'both':
        return mask, boundary
    else:  # assume 'mask'
        return mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', type=str, help='the input image file')
    parser.add_argument('-c', '--color-map', type=str, default='gray',
                        help='color map to be used')
    args = parser.parse_args()
    arr = cv2.imread(args.image_file, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    arr = arr / arr.max()
    mask = polymask(arr, win_title='polymask demo', return_type='mask',
                    color_map=args.color_map)
    plt.imshow(arr*mask, cmap=args.color_map)
    plt.show()
