import numpy as np

class GuidedFilter():
    def __init__(self, source, reference, r=64, eps= 0.05**2):
        self.source = source;
        self.reference = reference;
        self.r = r
        self.eps = eps

        self.smooth = self.guidedfilter(self.source,self.reference,self.r,self.eps)

    def boxfilter(self,img, r):
        (rows, cols) = img.shape
        imDst = np.zeros_like(img)

        imCum = np.cumsum(img, 0)
        imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
        imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
        imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]

        imCum = np.cumsum(imDst, 1)
        imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
        imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
        imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]

        return imDst

    def guidedfilter(self,I, p, r, eps):
        (rows, cols) = I.shape
        N = self.boxfilter(np.ones([rows, cols]), r)

        meanI = self.boxfilter(I, r) / N
        meanP = self.boxfilter(p, r) / N
        meanIp = self.boxfilter(I * p, r) / N
        covIp = meanIp - meanI * meanP

        meanII = self.boxfilter(I * I, r) / N
        varI = meanII - meanI * meanI

        a = covIp / (varI + eps)
        b = meanP - a * meanI

        meanA = self.boxfilter(a, r) / N
        meanB = self.boxfilter(b, r) / N

        q = meanA * I + meanB
        return q