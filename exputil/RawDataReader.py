#!/usr/bin/env python

import numpy as np


class PinducerRawDataReader:
    '''
    A class for reading raw data from both so-called PAW system and G2/G3
    system.
    '''

    SUPPORTED_FIPETYPE = ['G2', 'PAW']

    def __init__(self, fileType):
        if fileType not in self.SUPPORTED_FIPETYPE:
            # print 'unsupported file type: ' + fileType
            raise StandardError('Unsupported file type ' + fileType)
        self.fileType = fileType
        self.data = None

    def readDir(self, dataDir):
        if self.fileType == 'PAW':
            paraFile = os.path.join(dataDir, 'Parameters.dat')
            dataFile = os.path.join(dataDir, 'PA.dat')
            self.read(paraFile, dataFile)
        elif self.fileType == 'G2':
            pass
        return self.data

    def read(self, paraFile, dataFile):
        if self.fileType == 'PAW':
            self.para = self._readPAWParaFile(paraFile)
            self.data = self._readPAWDataFile(dataFile, self.para)
        elif self.fileType == 'G2':
            self.para = self._readG2ParaFile(paraFile)
            self.data = self._readG2DataFile(dataFile, self.para)
        return self.data

    @staticmethod
    def _readG2ParaFile(paraFile):
        para = dict()
        return para

    @staticmethod
    def _readG2DataFile(dataFile, para):
        pass

    @staticmethod
    def _readPAWParaFile(paraFile):
        para = dict()
        with open(paraFile) as f:
            lines = f.readlines()
            items = lines[0].split()
            for i in xrange(0, len(items), 2):
                key = items[i].lower()
                val = items[i+1]
                if key != 'name':
                    val = int(val)
                para[key] = val
        return para

    @staticmethod
    def _readPAWDataFile(dataFile, para):
        data = np.fromfile(dataFile, dtype=np.int16)
        data = np.reshape(data,
                          [para['datalength'], para['points'], para['lines']],
                          order='F')
        return data


class PinducerRawDataProcess:
    '''
    A class of static methods for raw data preprocess
    '''

    @staticmethod
    def preprocess(rawData, dcCorrection=True, flatten=False, returnStd=False):
        dataLength, points, lines = rawData.shape
        if dcCorrection:
            dcEstimate = np.mean(rawData, axis=0)
            data = rawData - dcEstimate
        if flatten:
            data = data.reshape((dataLength, points*lines), order='F')
        averagedData = np.mean(data, axis=1)
        if returnStd:
            # ddof = 1, "unbiased" estimate
            stdData = np.std(data, axis=1, ddof=1)
            return (averagedData, stdData)
        else:
            return averagedData


import argh
import os.path
from scipy.io import savemat


@argh.arg('file_type', choices=PinducerRawDataReader.SUPPORTED_FIPETYPE)
def main(data_dir, file_type, target_file):
    data_dir = os.path.abspath(data_dir)
    reader = PinducerRawDataReader(file_type)
    data = reader.readDir(data_dir)
    print 'Saving data to: ' + target_file
    savemat(target_file, {'data': data})

if __name__ == '__main__':
    argh.dispatch_command(main)
