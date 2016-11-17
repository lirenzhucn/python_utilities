'''additional i/o support functions'''

import vtk
import numpy as np
import skimage.io._plugins.freeimage_plugin as fi
import utils


def vtkDTypeToNumpyDType(dataType):
    if dataType == vtk.VTK_UNSIGNED_CHAR:
        return np.uint8
    elif dataType == vtk.VTK_SHORT:
        return np.int16
    elif dataType == vtk.VTK_UNSIGNED_CHAR:
        return np.uint16
    elif dataType == vtk.VTK_INT:
        return np.int32
    elif dataType == vtk.VTK_UNSIGNED_INT:
        return np.uint32
    elif dataType == vtk.VTK_LONG:
        return np.int64
    elif dataType == vtk.VTK_UNSIGNED_LONG:
        return np.uint64
    elif dataType == vtk.VTK_FLOAT:
        return np.float32
    elif dataType == vtk.VTK_DOUBLE:
        return np.float64
    else:
        return None


def loadVtk3d(filepath):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    imgVtk = reader.GetOutput()
    dataExport = vtk.vtkImageExport()
    dataExport.SetInputData(imgVtk)
    nx, ny, nz = dataExport.GetDataDimensions()
    dataType = dataExport.GetDataScalarType()
    npDataType = vtkDTypeToNumpyDType(dataType)
    if npDataType is None:
        # meaning data type not supported
        return (None, None)
    img = np.zeros((nx, ny, nz), dtype=npDataType, order='F')
    dataExport.Export(img)
    return (img, dataExport.GetDataSpacing())


def saveVtk3d(filepath, img, spacing=[1.0, 1.0, 1.0]):
    assert(img.ndim == 3)
    nx, ny, nz = img.shape
    dataImport = vtk.vtkImageImport()
    dataImport.SetNumberOfScalarComponents(1)
    dataImport.SetDataExtent(0, nx-1, 0, ny-1, 0, nz-1)
    dataImport.SetWholeExtent(0, nx-1, 0, ny-1, 0, nz-1)
    dataType = img.dtype
    if dataType == np.float_ or dataType == np.float64:
        dataImport.SetDataScalarTypeToDouble()
    elif dataType == np.float32:
        dataImport.SetDataScalarTypeToFloat()
    elif dataType == np.int_ or dataType == np.int32:
        dataImport.SetDataScalarTypeToInt()
    elif dataType == np.int16:
        dataImport.SetDataScalarTypeToShort()
    elif dataType == np.uint16:
        dataImport.SetDataScalarTypeToUnsignedShort()
    elif dataType == np.uint8:
        dataImport.SetDataScalarTypeToUnsignedChar()
    else:
        # convert image to unsigned short
        img = utils.nnormalize(img)
        img = np.round(img * 65535.0).astype(np.uint16)
    dataString = img.tostring(order='F')
    dataImport.CopyImportVoidPointer(dataString, len(dataString))
    dataImport.SetDataSpacing(spacing)
    dataImport.Update()
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(dataImport.GetOutput())
    writer.Write()


def loadTiff3d(filepath):
    imglist = fi.read_multipage(filepath)
    if imglist is None:
        return None
    nz = len(imglist)
    nx, ny = imglist[0].shape
    img = np.zeros((nx, ny, nz), dtype=imglist[0].dtype)
    for idx in xrange(nz):
        img[:, :, idx] = imglist[idx]
    return img


def saveTiff3d(filepath, img):
    nx, ny, nz = img.shape
    imglist = [img[:, :, idx] for idx in xrange(nz)]
    fi.write_multipage(imglist, filepath)
