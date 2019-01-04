import pyfits
import plasticc

datadir='/Users/suzuki/github/projects_plasticc/data_original/'
datadir='/work200t/nsuzuki/plasticc/data/'

def write_fitstable_ltcv():
    pl=plasticc.LSSTplasticc()
    #pl.read_ltcvdata('../data/training_set.csv')
    #pl.read_ltcvdata(datadir+'/test_set.csv')
    pl.read_ltcvdata(datadir+'/training_set.csv')
    #pl.read_ltcvdata_genfromtxt(datadir+'/test_set100.csv')
    #pl.read_ltcvdata_genfromtxt(datadir+'/test_set.csv')
    print('Finished Reading csv, Number of Object=',len(pl.ltcv['object_id']))

    c1=pyfits.Column(name='object_id',format='J',array=pl.ltcv['object_id'])
    print('mjd',len(pl.ltcv['mjd']))
    c2=pyfits.Column(name='mjd',format='E',array=pl.ltcv['mjd'])
    c3=pyfits.Column(name='passband',format='I',array=pl.ltcv['passband'])
    # print('flux',len(pl.ltcv['flux']))
    c4=pyfits.Column(name='flux',format='E',array=pl.ltcv['flux'])
    c5=pyfits.Column(name='flux_err',format='E',array=pl.ltcv['flux_err'])
    c6=pyfits.Column(name='detected',format='L',array=pl.ltcv['detected'])
    coldefs=pyfits.ColDefs([c1,c2,c3,c4,c5,c6])
    tblhdu=pyfits.BinTableHDU.from_columns(coldefs)
    #tblhdu.writeto('training_set.fits')
    #tblhdu.writeto(datadir+'/test_set.fits')
    #tblhdu.writeto('test_set100.fits')
    #tblhdu.writeto(datadir+'test_set.fits')
    tblhdu.writeto(datadir+'training_set.fits')

write_fitstable_ltcv()
