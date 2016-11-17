"""Provide a cropbox function to display a 2D image and let user to select a
rectangule ROI.
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import numpy as np
from matplotlib import cm


def _extract_pos_size(pos, size, im_shape):
    pos = pos
    size = size
    lefttop = list(pos)
    rightbot = [pos[0]+size[0]-1, pos[1]+size[1]-1]
    for pi in range(len(lefttop)):
        if lefttop[pi] < 0:
            lefttop[pi] = 0
        elif lefttop[pi] >= im_shape[pi]:
            lefttop[pi] = im_shape[pi]-1
    for pi in range(len(rightbot)):
        if rightbot[pi] < 0:
            rightbot[pi] = 0
        elif rightbot[pi] >= im_shape[pi]:
            rightbot[pi] = im_shape[pi]-1
    pos = lefttop
    size = [rb-lt+1 for rb, lt in zip(rightbot, lefttop)]
    pos = [round(p) for p in reversed(pos)]
    size = [round(s) for s in reversed(size)]
    return pos, size


def _compute_init_rect(im_shape):
    xsize, ysize = im_shape
    x0 = xsize // 4
    x1 = x0 + xsize // 2
    y0 = ysize // 4
    y1 = y0 + ysize // 2
    return ((x0, y0), (x1-x0, y1-y0))


def _convert_color(im, color_map, normalize=True):
    im = im.astype(np.float64)
    if normalize:
        im = im / im.max()
    cmObj = cm.get_cmap(color_map)
    return cmObj(im)


def _bound_index(ind, lower, upper):
    if ind < lower:
        ind = lower
    if ind > upper:
        ind = upper
    return ind


def _blend_images(bg_rgb, fg_rgb, fg_alpha, fg_pos, return_fg_slice=False):
    bg_rgb = np.copy(bg_rgb)
    bg_shape = bg_rgb.shape[:2]
    fg_shape = fg_rgb.shape[:2]
    if fg_alpha is not None:
        assert(fg_alpha.shape == fg_shape)
    bg_x0, bg_y0 = fg_pos
    bg_x1, bg_y1 = bg_x0 + fg_shape[0], bg_y0 + fg_shape[1]
    fg_x0 = 0
    if bg_x0 < 0:
        fg_x0 = -bg_x0
        bg_x0 = 0
    elif bg_x0 >= bg_shape[0]:
        bg_x0 = bg_shape[0] - 1
    fg_x1 = fg_shape[0]
    if bg_x1 > bg_shape[0]:
        bg_x1 = bg_shape[0]
        fg_x1 = bg_x1 - bg_x0 + fg_x0
    elif bg_x1 < 0:
        bg_x1 = 0
        fg_x0 = 0
        fg_x1 = 0
    fg_y0 = 0
    if bg_y0 < 0:
        fg_y0 = -bg_y0
        bg_y0 = 0
    elif bg_y0 >= bg_shape[1]:
        bg_y0 = bg_shape[1] - 1
    fg_y1 = fg_shape[1]
    if bg_y1 > bg_shape[1]:
        bg_y1 = bg_shape[1]
        fg_y1 = bg_y1 - bg_y0 + fg_y0
    elif bg_y1 < 0:
        bg_y1 = 0
        fg_y0 = 0
        fg_y1 = 0
    bg_slice = [slice(bg_x0, bg_x1, None), slice(bg_y0, bg_y1, None),
                slice(None)]
    fg_slice = [slice(fg_x0, fg_x1, None), slice(fg_y0, fg_y1, None)]
    if fg_alpha is not None:
        bg_rgb[bg_slice] *= (1 - fg_alpha[fg_slice + [np.newaxis]])
        bg_rgb[bg_slice] += (fg_alpha[fg_slice + [np.newaxis]] *
                             fg_rgb[fg_slice + [slice(None)]])
    else:
        bg_rgb[bg_slice] += fg_rgb[fg_slice + [slice(None)]]
        bg_rgb[bg_rgb > 1.0] = 1.0
    if return_fg_slice:
        return bg_rgb, fg_slice
    else:
        return bg_rgb


def make_slice(pos, size):
    res = [slice(p, p+s, None) for p, s in zip(pos, size)]
    return res


def place_subimage_and_cropbox(
        bg, fg, fg_alpha, bg_color_map='gray', fg_color_map='hot',
        win_title='place sub-image', crop_pos=None, crop_size=None,
        fg_pos=None, crop_pen=('r',), fg_pen=('b',)):
    """Place a foreground image `fg` on a background image `bg` interactive, while
    also provide a changeable ROI for croping of the background image.

    :param bg: background image
    :param fg: foreground image
    :param fg_alpha: foreground transparency channel
    :param bg_color_map: color map for background
    :param fg_color_map: color map for foreground
    :param win_title: the window's title
    :param crop_pos: initial position for the crop box (None means auto)
    :param crop_size: initial size of the crop box (None means auto)
    :param fg_pos: initial position of the foreground
    :param crop_pen: pen options for crop box
    :param fg_pen: pen options for foreground image
    """
    assert(bg.ndim == 2)
    assert(fg.ndim == 2)
    bg = bg.T
    fg = fg.T
    if fg_alpha is not None:
        fg_alpha = fg_alpha.T
    bgRgb = _convert_color(bg, bg_color_map)
    fgRgb = _convert_color(fg, fg_color_map)
    # default position and size
    defaultPos, defaultSize = _compute_init_rect(bg.shape)
    if crop_pos is None:
        crop_pos = defaultPos
    else:
        crop_pos = tuple(reversed(crop_pos))
    if crop_size is None:
        crop_size = defaultSize
    else:
        crop_size = tuple(reversed(crop_size))
    if fg_pos is None:
        fg_pos = defaultPos
    else:
        fg_pos = tuple(reversed(fg_pos))
    # initialize GUI stuffs
    win_size = [e + 10 for e in bg.shape]
    app = QtGui.QApplication([])
    widget = pg.GraphicsWindow(size=win_size, border=True)
    widget.setWindowTitle(win_title)
    widgetLayout = widget.addLayout(row=0, col=0)
    view = widgetLayout.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
    imItem = pg.ImageItem(_blend_images(bgRgb, fgRgb, fg_alpha, fg_pos))
    view.addItem(imItem)
    view.disableAutoRange('xy')
    view.autoRange()
    cropRoi = pg.RectROI(crop_pos, crop_size, pen=pg.mkPen(*crop_pen))
    view.addItem(cropRoi)
    fgRoi = pg.ROI(fg_pos, fg.shape, pen=pg.mkPen(*fg_pen))
    view.addItem(fgRoi)
    fg_slice = (slice(None), slice(None))

    def fg_move(roi):
        fg_pos = tuple(round(p) for p in roi.pos())
        im_rgb, fg_slice = _blend_images(bgRgb, fgRgb, fg_alpha, fg_pos,
                                         return_fg_slice=True)
        imItem.setImage(im_rgb)
        widget.update()

    fgRoi.sigRegionChangeFinished.connect(fg_move)
    app.exec_()
    crop_pos, crop_size = _extract_pos_size(cropRoi.pos(), cropRoi.size(),
                                            bg.shape)
    fg_pos, fg_size = _extract_pos_size(fgRoi.pos(), fgRoi.size(), bg.shape)
    crop_slice = tuple(slice(p, p+s) for p, s in zip(crop_pos, crop_size))
    return crop_slice, fg_slice, fg_pos, crop_pos, crop_size


def cropbox(im, win_title='cropbox', color_map='gray', pos=None, size=None,
            num_boxes=1, pens=(('r',))):
    """
    The main interface function.

    :param im: the input image, assumed 2D
    :param win_title: the string you want to appear as the window's title
    :param color_map: the color map you want the image to be displayed in
    :param pen: pen option passed to ROI object
    """
    assert(im.ndim == 2)
    im = im.T
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
    defaultPos, defaultSize = _compute_init_rect(im.shape)
    if pos is None:
        pos = [defaultPos] * num_boxes
    else:
        pos = [tuple(reversed(p)) for p in pos]
    if size is None:
        size = [defaultSize] * num_boxes
    else:
        size = [tuple(reversed(s)) for s in size]
    # create all the ROIs
    roiList = []
    for i in range(num_boxes):
        roi = pg.RectROI(pos[i], size[i], pen=pg.mkPen(*(pens[i])))
        view.addItem(roi)
        roiList.append(roi)
    app.exec_()
    posList = []
    sizeList = []
    for roi in roiList:
        # return the box
        pos, size = _extract_pos_size(roi.pos(), roi.size(), im.shape)
        posList.append(pos)
        sizeList.append(size)
    if num_boxes == 1:
        return (posList[0], sizeList[0])
    else:
        return (posList, sizeList)


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
    pos, size = cropbox(arr, win_title='cropbox demo',
                        color_map=args.color_map)
    plt.imshow(arr[make_slice(pos, size)], cmap=args.color_map)
    plt.show()
