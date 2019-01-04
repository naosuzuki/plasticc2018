import numpy
import pyfits
import time
import sys

datadir='/Users/suzuki/github/projects_plasticc/data_original/'
datadir='/work200t/nsuzuki/plasticc/data/'

'''
2018-12-01 : Nao Suzuki
Reading Plasticc FITS Table
'''

def read_fitstable_ltcv():
    start=time.time()
    #ltcvtbl=pyfits.open('test_set100.fits',memmap=True)
    #ltcvtbl=pyfits.open('test_set100.fits',memmap=True)
    ltcvtbl=pyfits.open(datadir+'test_set.fits',memmap=True)
    ltcvdata=ltcvtbl[1].data
    #rows=numpy.arange(20)
    #print(ltcvdata[rows])
    print(ltcvdata[-3])
    print(ltcvdata[-2])
    print(ltcvdata[-1])

read_fitstable_ltcv()
