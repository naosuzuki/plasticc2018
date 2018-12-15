import numpy
import matplotlib
import plasticc
import sys
import scipy
from scipy import stats
import pandas as pd
import pyfits
import fitsio
import time

#datadir='/work/nsuzuki/plasticc2018/data/'
datadir='/Users/suzuki/github/projects_plasticc/data_original/'
targetclass=numpy.array([6,15,16,42,52,53,62,64,65,67,88,90,92,95])
#object_id,ra,decl,gal_l,gal_b,ddf,hostgal_specz,hostgal_photoz,hostgal_photoz_err,distmod,mwebv,target

def write_fitstable_metadata():
    pl=plasticc.LSSTplasticc()
    #test_set = pd.read_csv('../data_original/test_set.csv')
    #pl.read_metadata('../data/training_set_metadata.csv')
    #pl.read_metadata('/work/nsuzuki/plasticc2018/data/test_set_metadata.csv')
    pl.read_metadata(datadir+'test_set_metadata.csv')
    print(len(pl.metadata['object_id']))
    c1=pyfits.Column(name='object_id',format='J',array=pl.metadata['object_id'])
    c2=pyfits.Column(name='ra',format='E',array=pl.metadata['ra'])
    c3=pyfits.Column(name='decl',format='E',array=pl.metadata['decl'])
    c4=pyfits.Column(name='gal_l',format='E',array=pl.metadata['gal_l'])
    c5=pyfits.Column(name='gal_b',format='E',array=pl.metadata['gal_b'])
    c6=pyfits.Column(name='ddf',format='L',array=pl.metadata['ddf'])
    c7=pyfits.Column(name='hostgal_specz',format='E',array=pl.metadata['hostgal_specz'])
    c8=pyfits.Column(name='hostgal_photoz',format='E',array=pl.metadata['hostgal_photoz'])
    c9=pyfits.Column(name='hostgal_photoz_err',format='E',array=pl.metadata['hostgal_photoz_err'])
    c10=pyfits.Column(name='distmod',format='E',array=pl.metadata['distmod'])
    c11=pyfits.Column(name='mwebv',format='E',array=pl.metadata['mwebv'])
    #c12=pyfits.Column(name='target',format='I',array=pl.metadata['target'])
    #coldefs=pyfits.ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12])
    coldefs=pyfits.ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11])
    tblhdu=pyfits.BinTableHDU.from_columns(coldefs)
    #tblhdu.writeto('training_set_metadata.fits')
    tblhdu.writeto('test_set_metadata.fits')

def read_fitstable_metadata():
    #pl=plasticc.LSSTplasticc()
    tbl=pyfits.open('../data/training_set_metadata.fits')
    metadata=tbl[1].data
    #print(metadata['object_id'])
    #print(metadata['ra'])
    #print(metadata['decl'])
    #print(metadata['hostgal_specz'])
    #print(metadata['distmod'])


def write_fitstable_ltcv():
    pl=plasticc.LSSTplasticc()
    #pl.read_ltcvdata('../data/training_set.csv')
    pl.read_ltcvdata(datadir+'/test_set.csv')
    print('Finished Reading csv, Number of Object=',len(pl.ltcv['object_id']))
    c1=pyfits.Column(name='object_id',format='J',array=pl.ltcv['object_id'])
    print('mjd',len(pl.ltcv['mjd']))
    c2=pyfits.Column(name='mjd',format='E',array=pl.ltcv['mjd'])
    c3=pyfits.Column(name='passband',format='I',array=pl.ltcv['passband'])
    print('flux',len(pl.ltcv['flux']))
    c4=pyfits.Column(name='flux',format='E',array=pl.ltcv['flux'])
    c5=pyfits.Column(name='flux_err',format='E',array=pl.ltcv['flux_err'])
    c6=pyfits.Column(name='detected',format='L',array=pl.ltcv['detected'])
    coldefs=pyfits.ColDefs([c1,c2,c3,c4,c5,c6])
    tblhdu=pyfits.BinTableHDU.from_columns(coldefs)
    #tblhdu.writeto('training_set.fits')
    tblhdu.writeto(datadir+'/test_set.fits')

def read_fitstable_ltcv():
    #pl=plasticc.LSSTplasticc()
    #tbl=pyfits.open('../data/training_set.fits',memmap=True)
    start=time.time()
    # Reading Meta data
    metatbl=pyfits.open(datadir+'test_set_metadata.fits',memmap=True)
    metadata=metatbl[1].data
    end=time.time()
    print('Meta data is read')
    print('Metadata readout time',end-start)

    start=time.time()
    # Reading LTCV data
    ltcvtbl=pyfits.open(datadir+'test_set.fits',memmap=True)
    ltcvdata=ltcvtbl[1].data
    end=time.time()
    print('LTCV data is read')
    print('LTCVdata readout time',end-start)
    #rows=numpy.where((ltcvdata['object_id']==objid) & (ltcvdata['passband']==1) & (ltcvdata['detected']==True))

    rows=numpy.arange(30)
    rows=rows+100000000

    start=time.time()
    ltcv1=ltcvdata[rows]
    rows1=numpy.where((ltcv1['passband']==1) & (ltcv1['detected']==True))
    ltcvdata1=ltcv1[rows1]
    end=time.time()
    print('extract 1',end-start)
    print(ltcvdata1)

    rows=rows+100000000
    start=time.time()
    ltcv2=ltcvdata[rows]
    end=time.time()
    print('extract 2',end-start)
    print(ltcv2)

    rows=rows+100000000
    start=time.time()
    ltcv3=ltcvdata[rows]
    end=time.time()
    print('extract 3',end-start)
    print(ltcv3)

    rows=rows+100000000
    start=time.time()
    ltcv4=ltcvdata[rows]
    end=time.time()
    print('extract 4',end-start)
    print(ltcv4)
    #print(metadata[10000])
    #print(metadata[10000]['object_id'])

    start=time.time()
    print("testing by object id")
    objid=metadata[10000]['object_id']
    rows=numpy.where((ltcvdata['object_id']==objid) & (ltcvdata['passband']==1) & (ltcvdata['detected']==True))
    ltcv5=ltcvdata[rows]
    end=time.time()
    print('by object_id extract 5',end-start)
    print(ltcv5)

    for i in range(10):
       print("testing by object id loop")
       start=time.time()
       objid=metadata[i]['object_id']
       rows=numpy.where((ltcvdata['object_id']==objid) & (ltcvdata['passband']==1) & (ltcvdata['detected']==True))
       ltcvx=ltcvdata[rows]
       end=time.time()
       print('by object_id extract x loop',i,end-start)
       print(ltcvx)

    metatbl.close()
    ltcvtbl.close()
    #print(ltcvdata['object_id'])
    #print(ltcvdata['flux'])
    #indivdata=ltcvdata['object_id'==745]
    #print(indivdata['flux'])
def read_fitstable_ltcv2():
    metadata=fitsio.read(datadir+'test_set_metadata.fits',memmap=True)
    print('Meta data is read')
    ltcvdata=fitsio.read(datadir+'test_set.fits')
    print('LTCV data is read')
    objid=metadata[10000]['object_id']
    rows=numpy.where((ltcvdata['object_id']==objid) & (ltcvdata['passband']==1) & (ltcvdata['detected']==True))
    ltcv1=ltcvdata[rows]
    print(ltcv1)
    
def read_asciitable_ltcv():
    #pl=plasticc.LSSTplasticc()
    #pl.read_metadata(datadir+'test_set_metadata.csv')
    metadata=pd.read_csv(datadir+'test_set_metadata.csv')
    print('Meta data is read')
    start=time.time()
    ltcvdata=pd.read_csv(datadir+'test_set.csv')
    print('LTCV data is read')
    end=time.time()
    print('readout time',end-start)
    #metadata['object_id']
    objid=metadata[10000]['object_id']
    #rows=numpy.where((ltcvdata['object_id']==objid) & (ltcvdata['passband']==1) & (ltcvdata['detected']==True))
    rows=numpy.arange(10)
    rows=rows+100000000
    start=time.time()
    ltcv1=ltcvdata[rows]
    end=time.time()
    print('extract 1',end-start)
    print(ltcv1)
    rows=rows+100000000
    start=time.time()
    ltcv2=ltcvdata[rows]
    end=time.time()
    print('extract 2',end-start)
    print(ltcv2)
    rows=rows+100000000
    start=time.time()
    ltcv3=ltcvdata[rows]
    end=time.time()
    print('extract 3',end-start)
    print(ltcv3)
    rows=rows+100000000
    start=time.time()
    ltcv4=ltcvdata[rows]
    end=time.time()
    print('extract 4',end-start)
    print(ltcv4)

#read_asciitable_ltcv()
read_fitstable_ltcv()
#read_fitstable_ltcv2()
#write_fitstable_metadata()
#read_fitstable_metadata()
#read_fitstable_ltcv()
#write_fitstable_ltcv()
