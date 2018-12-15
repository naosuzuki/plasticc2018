import numpy
import pyfits
import time
import sys

datadir='/Users/suzuki/github/projects_plasticc/data_original/'

'''
2018-12-01 : Nao Suzuki
Reading Plasticc FITS Table
'''

def read_fitstable_ltcv():
    start=time.time()
    ltcvtbl=pyfits.open('test_set100.fits',memmap=True)
    ltcvdata=ltcvtbl[1].data
    print(ltcvdata)

read_fitstable_ltcv()
