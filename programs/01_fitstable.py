import numpy
import matplotlib
import plasticc
import sys
import scipy
from scipy import stats
import pandas as pd
import pyfits

targetclass=numpy.array([6,15,16,42,52,53,62,64,65,67,88,90,92,95])
#object_id,ra,decl,gal_l,gal_b,ddf,hostgal_specz,hostgal_photoz,hostgal_photoz_err,distmod,mwebv,target

def write_fitstable_metadata():
    pl=plasticc.LSSTplasticc()
    #test_set = pd.read_csv('../data_original/test_set.csv')
    pl.read_metadata('../data/training_set_metadata.csv')
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
    c12=pyfits.Column(name='target',format='I',array=pl.metadata['target'])
    coldefs=pyfits.ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12])
    tblhdu=pyfits.BinTableHDU.from_columns(coldefs)
    tblhdu.writeto('training_set_metadata.fits')

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
    pl.read_ltcvdata('../data/training_set.csv')
    print(len(pl.ltcv['object_id']))
    c1=pyfits.Column(name='object_id',format='J',array=pl.ltcv['object_id'])
    c2=pyfits.Column(name='mjd',format='E',array=pl.ltcv['mjd'])
    c3=pyfits.Column(name='passband',format='I',array=pl.ltcv['passband'])
    c4=pyfits.Column(name='flux',format='E',array=pl.ltcv['flux'])
    c5=pyfits.Column(name='flux_err',format='E',array=pl.ltcv['flux_err'])
    c6=pyfits.Column(name='detected',format='L',array=pl.ltcv['detected'])
    coldefs=pyfits.ColDefs([c1,c2,c3,c4,c5,c6])
    tblhdu=pyfits.BinTableHDU.from_columns(coldefs)
    tblhdu.writeto('training_set.fits')

def read_fitstable_ltcv():
    pl=plasticc.LSSTplasticc()
    tbl=pyfits.open('../data/training_set.fits')
    metadata=tbl[1].data
    print(metadata['object_id'])
    print(metadata['flux'])


#write_fitstable_metadata()
#read_fitstable_metadata()
read_fitstable_ltcv()
#write_fitstable_ltcv()
