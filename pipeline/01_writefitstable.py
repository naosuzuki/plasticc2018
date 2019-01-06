import pyfits
import plasticc

datadir='/Users/suzuki/github/projects_plasticc/data_original/'
<<<<<<< HEAD
datadir='/work200t/nsuzuki/plasticc/data/'
=======
datadir='/work/nsuzuki/plasticc2018/data/'
>>>>>>> 28b73aac5acf30c9571db9bfad3da0397471c555

def write_fitstable_ltcv():
    pl=plasticc.LSSTplasticc()
    #pl.read_ltcvdata('../data/training_set.csv')
<<<<<<< HEAD
    #pl.read_ltcvdata(datadir+'/test_set.csv')
    pl.read_ltcvdata(datadir+'/training_set.csv')
    #pl.read_ltcvdata_genfromtxt(datadir+'/test_set100.csv')
=======
<<<<<<< HEAD
    #pl.read_ltcvdata_genfromtxt(datadir+'/test_set100.csv')
    pl.read_ltcvdata(datadir+'/test_set.csv')
=======
    pl.read_ltcvdata(datadir+'/test_set.csv')
    #pl.read_ltcvdata_genfromtxt(datadir+'/test_set100.csv')
>>>>>>> 35ecb853368670c84dd005843c1af36a1c9e5297
>>>>>>> 28b73aac5acf30c9571db9bfad3da0397471c555
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
<<<<<<< HEAD
    #tblhdu.writeto(datadir+'test_set.fits')
    tblhdu.writeto(datadir+'training_set.fits')
=======
    tblhdu.writeto('test_set_v2.fits')
>>>>>>> 28b73aac5acf30c9571db9bfad3da0397471c555

write_fitstable_ltcv()
