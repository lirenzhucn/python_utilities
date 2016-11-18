'''Visualization module'''

# import vtk
import numpy as np


def vtkMip3d(img8bit):
    nx, ny, nz = img8bit.shape
    dataImport = vtk.vtkImageImport()
    dataString = img8bit.tostring(order='F')
    dataImport.CopyImportVoidPointer(dataString, len(dataString))
    dataImport.SetDataScalarTypeToUnsignedChar()
    dataImport.SetDataExtent(0, nx-1, 0, ny-1, 0, nz-1)
    dataImport.SetWholeExtent(0, nx-1, 0, ny-1, 0, nz-1)
    alphaFunc = vtk.vtkPiecewiseFunction()
    alphaFunc.AddPoint(0, 0.0)
    alphaFunc.AddPoint(255, 1.0)
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0)
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(colorFunc)
    prop.SetScalarOpacity(alphaFunc)
    mapper = vtk.vtkSmartVolumeMapper()
    # mapper.SetInputConnection(dataImport.GetOutputPort())
    mapper.SetBlendModeToMaximumIntensity()
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    dataImport.Update()
    mapper.SetInputData(dataImport.GetOutput())
    # rendering
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(renWin)
    ren.AddVolume(volume)
    ren.SetBackground(0, 0, 0)
    ren.GetActiveCamera().ParallelProjectionOn()
    ren.ResetCamera()
    renWin.SetSize(400, 400)
    iren.Initialize()
    renWin.Render()
    iren.Start()


def mip3d(img, v_range=None):
    '''MIP rendering with a gray colormap'''
    assert(img.ndim == 3)
    if v_range is None:
        v_min = np.min(img)
        v_max = np.max(img)
    else:
        v_min = v_range[0]
        v_max = v_range[1]
    img = img.astype(np.float)
    scaled = (img - v_min) / (v_max - v_min)
    scaled[scaled < 0.0] = 0.0
    scaled[scaled > 1.0] = 1.0
    img8bit = (scaled * 255.0).astype(np.uint8)
    vtkMip3d(img8bit)


from .ImageSliceDisplay import ImageSliceDisplay
from PyQt5.QtWidgets import QApplication

# place holder for qt application
qApp = None


def _create_qApp():
    global qApp
    if qApp is None:
        app = QApplication.instance()
        if app is None:
            qApp = QApplication([])
            qApp.lastWindowClosed.connect(qApp.quit)
        else:
            qApp = app
    return qApp


def imshow3d(img, cmapName='gray', vrange=None,
             return_image=False, return_min_max=False, return_tgc=False):
    '''imshow:
    Display a 2D or 3D image with a matplotlib predefined colormap
    in the Qt-powered ImageSliceDisplay widget.
    '''
    # app = _create_qApp()
    app = QApplication([])
    # app.setStyle('plastique')
    app.setStyle('windows')
    widget = ImageSliceDisplay()
    widget.setWindowTitle('Image Slice Display')
    widget.setInput(img, cmapName)
    if vrange is not None:
        widget.setRange(*vrange)
    widget.show()
    app.exec_()
    returnList = []
    if return_image:
        returnList.append(widget.imgData)
    if return_min_max:
        returnList.append((float(widget.dMin), float(widget.dMax)))
    if return_tgc:
        returnList.append(widget.tgcDialog.TGCVector)
    if len(returnList) == 1:
        return returnList[0]
    elif len(returnList) > 1:
        return returnList
