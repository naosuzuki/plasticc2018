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
<<<<<<< HEAD
    #ltcvtbl=pyfits.open('test_set100.fits',memmap=True)
    ltcvtbl=pyfits.open(datadir+'test_set.fits',memmap=True)
    ltcvdata=ltcvtbl[1].data
    #rows=numpy.arange(20)
    #print(ltcvdata[rows])
    print(ltcvdata[-3])
    print(ltcvdata[-2])
    print(ltcvdata[-1])
=======
    #ltcvtbl=pyfits.open('test_set.fits',memmap=True)
    ltcvtbl=pyfits.open('test_set.fits')
    ltcvdata=ltcvtbl[1].data
    rows=numpy.arange(50)
    print(ltcvdata[rows])
>>>>>>> 28b73aac5acf30c9571db9bfad3da0397471c555

read_fitstable_ltcv()
