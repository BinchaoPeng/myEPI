import numpy as np
import os


class DataInfo:
    def __init__(self, filename):
        self.seqLenList = []
        self.data = np.load(filename)
        self.dataLen = len(self.data)

    def getItemSeqLen(self):
        for d in self.data:
            self.seqLenList.append(len(d))


if __name__ == '__main__':
    dataInfo = DataInfo("../../data/enhancer_of_K562.npy")
    print(dataInfo.dataLen)
    dataInfo.getItemSeqLen()
    print(dataInfo.seqLenList)
