import numpy
import scipy
from scipy import stats
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import pyfits
import cosmology

#object_id,ra,decl,gall,galb,ddf_bool,hostgal_specz,hostgal_photoz,hostgal_photoz_err,distmod,mwebv,target

targetclass=numpy.array([6,15,16,42,52,53,62,64,65,67,88,90,92,95])
extragalactictarget=numpy.array([15,42,52,62,64,67,88,90,95]) 
galactictarget=numpy.array([6,16,53,65,92])

#ddfdir='../ltcv_training/ddf/'
#widedir='../ltcv_training/wide/'

class LSSTplasticc:
# Script for Plasticc Challenge
# Written by Nao Suzuki (Kavli IPMU)

      def __init__(self):
          self.object_id=1
          self.ra=0.0
          self.decl=0.0
          self.gall=0.0
          self.galb=0.0
          self.ddf_bool=0
          self.hostgal_specz=0.0
          self.hostgal_photoz=0.0
          self.hostgal_photoz_err=0.0
          self.distmod=99.0
          self.mwebv=0.0
          self.target=0

      def read_metadata(self,csvfilename):
          self.metadata = pd.read_csv(csvfilename)
          self.object_id_list=self.metadata['object_id']
          self.ra_list=self.metadata['ra']
          self.decl_list=self.metadata['decl']
          self.gall_list=self.metadata['gal_l']
          self.galb_list=self.metadata['gal_b']
          self.ddf_list=self.metadata['ddf']
          self.specz_list=self.metadata['hostgal_specz']
          self.photoz_list=self.metadata['hostgal_photoz']
          self.photozerr_list=self.metadata['hostgal_photoz_err']
          #self.metadata[numpy.isnan(self.metadata['distmod'])]=0
          self.dm_list=self.metadata['distmod']
          self.mwebv_list=self.metadata['mwebv']
          # Training Set or Test Set
          if(csvfilename.find('training')!=-1):
             self.target_list=self.metadata['target']

      def read_metadatafits(self,metafitsfilename):
          metatbl=pyfits.open(metafitsfilename,memmap=True)
          self.metadata=metatbl[1].data 
          self.object_id_list=self.metadata['object_id']
          self.ra_list=self.metadata['ra']
          self.decl_list=self.metadata['decl']
          self.gall_list=self.metadata['gal_l']
          self.galb_list=self.metadata['gal_b']
          self.ddf_list=self.metadata['ddf']
          self.specz_list=self.metadata['hostgal_specz']
          self.photoz_list=self.metadata['hostgal_photoz']
          self.photozerr_list=self.metadata['hostgal_photoz_err']
          #self.metadata[numpy.isnan(self.metadata['distmod'])]=0
          self.dm_list=self.metadata['distmod']
          self.mwebv_list=self.metadata['mwebv']
          # Training Set or Test Set
          if(metafitsfilename.find('training')!=-1):
             self.target_list=self.metadata['target']
          if(metafitsfilename.find('plus')!=-1):
             self.start_row_list=self.metadata['start_row']
             self.end_row_list=self.metadata['end_row']

      def read_ltcvdatafits(self,ltcvfitsfilename):
          # Reading LTCV data
          ltcvtbl=pyfits.open(ltcvfitsfilename,memmap=True)
          self.ltcvdata=ltcvtbl[1].data
          self.ltcv_id_list=self.ltcvdata['object_id']
          self.ltcv_mjd_list=self.ltcvdata['mjd']
          self.ltcv_filter_list=self.ltcvdata['passband']
          self.ltcv_flux_list=self.ltcvdata['flux']
          self.ltcv_fluxerr_list=self.ltcvdata['flux_err']
          self.ltcv_detected_list=self.ltcvdata['detected']

      def extract_ltcvfits(self,objectid):
          [rows]=numpy.where(self.object_id_list==objectid)
          self.object_id=objectid
          [self.ddf]=self.ddf_list[rows]
          [self.z]=self.photoz_list[rows]
          [self.zerr]=self.photozerr_list[rows]
          if(self.specz_list[rows]>0.0):
             [self.z]=self.specz_list[rows]
             self.zerr=0.0
          [self.dm]=self.dm_list[rows]
          [self.mwebv]=self.mwebv_list[rows]
          #Metadata
          [self.start_row]=self.metadata['start_row'][rows]
          [self.end_row]=self.metadata['end_row'][rows]
          #LTCVdata
          self.ltcv=self.ltcvdata[self.start_row:self.end_row]
          self.mjd=self.ltcv_mjd_list[self.start_row:self.end_row]
          self.filter=self.ltcv_filter_list[self.start_row:self.end_row]
          self.flux=self.ltcv_flux_list[self.start_row:self.end_row]
          self.fluxerr=self.ltcv_fluxerr_list[self.start_row:self.end_row]
          self.detected=self.ltcv_detected_list[self.start_row:self.end_row]

      def extract_ltcvfitsbyfilter(self):
          frows0=numpy.where(self.filter==0)
          frows1=numpy.where(self.filter==1)
          frows2=numpy.where(self.filter==2)
          frows3=numpy.where(self.filter==3)
          frows4=numpy.where(self.filter==4)
          frows5=numpy.where(self.filter==5)
          self.mjd0=self.mjd[frows0] ; self.flux0=self.flux[frows0] ; self.fluxerr0=self.fluxerr[frows0]
          self.mjd1=self.mjd[frows1] ; self.flux1=self.flux[frows1] ; self.fluxerr1=self.fluxerr[frows1]
          self.mjd2=self.mjd[frows2] ; self.flux2=self.flux[frows2] ; self.fluxerr2=self.fluxerr[frows2]
          self.mjd3=self.mjd[frows3] ; self.flux3=self.flux[frows3] ; self.fluxerr3=self.fluxerr[frows3]
          self.mjd4=self.mjd[frows4] ; self.flux4=self.flux[frows4] ; self.fluxerr4=self.fluxerr[frows4]
          self.mjd5=self.mjd[frows5] ; self.flux5=self.flux[frows5] ; self.fluxerr5=self.fluxerr[frows5]

      def extract_basics(self):
          # Number of Data Points
          self.num=len(self.mjd)
          # Number of Detected Data Points
          [drows]=numpy.where(self.detected==True)
          #[drows_opt]=numpy.where((self.detected==True) & (self.filter!=5))

          self.dnum=len(drows)
          # Event Duration
          if(self.dnum>2):
             self.dtime=(self.mjd[drows[-1]]-self.mjd[drows[0]])/(1.0+self.z)
             #self.mjdmaxbyflux=numpy.sum(self.flux[drows]*(self.mjd[drows]-self.mjd[0]))/numpy.sum(self.flux[drows])/(1.0+self.z)
             self.mjdmaxbyflux=numpy.sum(self.flux[drows]*(self.mjd[drows]-self.mjd[drows[0]]))/numpy.sum(self.flux[drows])/(1.0+self.z)
          else:
             self.dtime=0.0
             self.mjdmaxbyflux=0.0
          # Flux Average
          self.fluxmean=numpy.mean(self.flux[drows])
          self.fluxstd=numpy.std(self.flux[drows])
          self.errmean=numpy.mean(self.fluxerr[drows])
          self.sigmamean=numpy.mean(self.flux[drows]/self.fluxerr[drows])
          self.sigmasum=numpy.sum(self.flux[drows]/self.fluxerr[drows])
          self.sigmastd=numpy.mean(self.flux[drows]/self.fluxerr[drows])
          self.skewness=scipy.stats.skew(self.flux[drows])
          self.kurtosis=scipy.stats.kurtosis(self.flux[drows])
          if(self.num!=0):
             self.detectionfraction=float(self.dnum)/float(self.num)

          #if(len(drows_opt)>0):
          #    self.fluxmax=max(self.flux[drows_opt])
          self.fluxmax=numpy.zeros(6,dtype=numpy.float)
          self.absmag=numpy.zeros(6,dtype=numpy.float)
          self.color=numpy.zeros(5,dtype=numpy.float)

          self.fluxmaxall=max(self.flux[drows])

          # Absolute Magnitude
          if((self.z>0.0)&(self.fluxmaxall>0.0)):
             lcdm=cosmology.Cosmology()
             dm=lcdm.DistMod(self.z)
             self.absmagall=27.5-2.5*numpy.log10(self.fluxmaxall)-dm
          elif((self.z==0.0) &(self.fluxmaxall>0.0)):
             self.absmagall=27.5-2.5*numpy.log10(self.fluxmaxall)
          else:
             self.absmagall=9.9

          for i in range(6):
             [drows_filter]=numpy.where((self.detected==True) & (self.filter==i))
             if(len(drows_filter)>0):
                 self.fluxmax[i]=max(self.flux[drows_filter])
                 if((self.fluxmax[i]>0.0) & (self.z>0.0)):
                    self.absmag[i]=27.5-2.5*numpy.log10(self.fluxmax[i])-dm
                 elif(self.fluxmax[i]>0.0):
                    self.absmag[i]=27.5-2.5*numpy.log10(self.fluxmax[i])

          # Color at Max
          for i in range(5):
              if((self.fluxmax[i]!=0.0) & (self.fluxmax[i+1]!=0.0) & (self.fluxmax[i]*self.fluxmax[i+1]>0.0)):
                 #print(i,self.fluxmax[i],self.fluxmax[i+1])
                 self.color[i]=-2.5*numpy.log10(self.fluxmax[i]/self.fluxmax[i+1])
              else:
                 self.color[i]=9.9

          flag=0
          if((flag==0) & (self.color[2]!=9.9)):
              flag=1  ; self.colorall=self.color[2]
          elif((flag==0) & (self.color[1]!=9.9)):
              flag=1  ; self.colorall=self.color[1]
          elif((flag==0) & (self.color[3]!=9.9)):
              flag=1  ; self.colorall=self.color[3]
          elif((flag==0) & (self.color[0]!=9.9)):
              flag=1  ; self.colorall=self.color[0]
          elif((flag==0) & (self.color[4]!=9.9)):
              flag=1  ; self.colorall=self.color[4]
           
      def read_ltcvdata(self,csvfilename):
          datatype={'object_id':'int32','mjd':'float32',\
          'passband':'int8','flux':'float16','flux_err':'float16',\
          'detected':'int8'}
          #self.ltcv = pd.read_csv(csvfilename,dtype=datatype,nrows=100)
          #self.ltcv = pd.read_csv(csvfilename,nrows=100)
          self.ltcv = pd.read_csv(csvfilename)
          self.ltcv_id_list=self.ltcv['object_id']
          self.ltcv_mjd_list=self.ltcv['mjd']
          self.ltcv_filter_list=self.ltcv['passband']
          self.ltcv_flux_list=self.ltcv['flux']
          self.ltcv_fluxerr_list=self.ltcv['flux_err']
          self.ltcv_flag_list=self.ltcv['detected']

      def read_ltcvdata_genfromtxt(self,csvfilename):
          ltcvfile=open(csvfilename,'r')
          ltcvdata=ltcvfile.readlines()
          npix=len(ltcvdata)
          ltcvfile.close()

          #object_id,mjd,passband,flux,flux_err,detected
          # f4=float 32 bits ; i4=integer 32 bits
          self.ltcv=numpy.genfromtxt(ltcvdata,dtype="i8,f4,i4,f4,f4,i4",\
          names=['object_id','mjd','passband','flux','flux_err','detected'],\
          comments='#',delimiter=',',skip_header=1)
          self.ltcv_id_list=self.ltcv[:]['object_id']
          self.ltcv_mjd_list=self.ltcv[:]['mjd']
          self.ltcv_filter_list=self.ltcv[:]['passband']
          self.ltcv_flux_list=self.ltcv[:]['flux']
          self.ltcv_fluxerr_list=self.ltcv[:]['flux_err']
          self.ltcv_flag_list=self.ltcv[:]['detected']

      def plot_trainingltcv(self,objectid):

          [lsstcolor,lsstfilter]=load_lsstdictionary()

          #self.extract_ltcv(objectid)
          self.read_trainingltcv(objectid)
          objname=str(objectid)
          if(self.ddf_bool==1): ddfdir='ddf'
          if(self.ddf_bool==0): ddfdir='wide'
          classdir='../ltcv_training/'+ddfdir+'/class_'+str(self.target)
          if not (os.path.exists(classdir)): os.mkdir(classdir)
          asciifiledir='../ltcv_training/'+ddfdir+'/class_'+str(self.target)+'/'+objname
          if not (os.path.exists(asciifiledir)): os.mkdir(asciifiledir)
          pngfilename='../ltcv_training/'+ddfdir+'/class_'+str(self.target)+'/'+objname+'/'+objname+'.png'
          #print(pngfilename)

          # Latex
          plt.figure(figsize=(8,6), dpi=100)
          plt.rc('text',usetex=True)
          # Font to be Times
          plt.rc('font',family='serif')
          plt.title('Plasticc Training ')
          plt.xlabel('MJD')
          plt.ylabel('Flux (Zpt=27.5)')
         
          xmin=numpy.min(self.ltcvmjd)-3.0
          xmax=numpy.max(self.ltcvmjd)+3.0
          #print(xmin,xmax)
          ymin=numpy.min(self.ltcvflux)*1.05
          ymax=numpy.max(self.ltcvflux)*1.05

          plt.xlim([xmin,xmax])
          plt.ylim([ymin,ymax])
          plt.plot([xmin,xmax],[0.0,0.0],linestyle='dotted',color='k')

          for i in range(6):
             self.flagfilter=numpy.where(self.ltcvfilter==i,1,0)
             self.mjd=numpy.compress(self.flagfilter,self.ltcvmjd)
             self.flux=numpy.compress(self.flagfilter,self.ltcvflux)
             self.fluxerr=numpy.compress(self.flagfilter,self.ltcvfluxerr)
             self.flag=numpy.compress(self.flagfilter,self.ltcvflag)
             plt.plot(self.mjd,self.flux,'o',color=lsstcolor[i],markersize=2.0,label=lsstfilter[i])
             del self.mjd ; del self.flux ; del self.fluxerr ; del self.flag
             #plt.errorbar(self.mjd,self.flux,yerr=self.fluxerr,\
             #fmt='o',mfc=lsstcolor[i],mec=lsstcolor[i],markerfacecoloralt=lsstcolor[i],markersize=0.2)

          plt.legend()
          plt.savefig(pngfilename,format='png')
          plt.clf()
          plt.close()

      def mkdir_class(self):
         if(os.path.exists(ddfdir)==False):
            print('hello ddf')
            os.mkdir(ddfdir)
         if(os.path.exists(widedir)==False):
            print('hello wide')
            os.mkdir(widedir)

         for i in range(len(targetclass)):
            if(os.path.exists(ddfdir+'/'+'object_'+str(targetclass[i]))==False):
               os.mkdir(ddfdir+'/'+'object_'+str(targetclass[i]))
            if(os.path.exists(widedir+'/'+'object_'+str(targetclass[i]))==False):
               os.mkdir(widedir+'/'+'object_'+str(targetclass[i]))

      def extract_ltcv(self,objectid):
          self.object_id=objectid
          #flaglist=numpy.where(self.ltcv_id_list==objectid)
          # Extract ObjectID
          metaflag=numpy.where(self.ltcv_id_list==objectid,1,0)
          self.mjdlist=numpy.compress(metaflag,self.ltcv_mjd_list)
          self.filterlist=numpy.compress(metaflag,self.ltcv_filter_list)
          self.fluxlist=numpy.compress(metaflag,self.ltcv_flux_list)
          self.fluxerrlist=numpy.compress(metaflag,self.ltcv_fluxerr_list)
          self.flaglist=numpy.compress(metaflag,self.ltcv_flag_list)

      def extract_metadata(self,objectid):
          self.object_id=objectid
          flaglist=numpy.where(self.object_id_list==objectid,1,0)
          [self.ra]=numpy.compress(flaglist,self.ra_list)
          [self.decl]=numpy.compress(flaglist,self.decl_list)
          [self.gall]=numpy.compress(flaglist,self.gall_list)
          [self.galb]=numpy.compress(flaglist,self.galb_list)
          [self.ddf_bool]=numpy.compress(flaglist,self.ddf_list)
          [self.host_specz]=numpy.compress(flaglist,self.specz_list)
          [self.host_photoz]=numpy.compress(flaglist,self.photoz_list)
          [self.host_photozerr]=numpy.compress(flaglist,self.photozerr_list)
          [self.distmod]=numpy.compress(flaglist,self.dm_list)
          [self.mwebv]=numpy.compress(flaglist,self.mwebv_list)
          [self.target]=numpy.compress(flaglist,self.target_list)

          #self.dmlist[numpy.isnan(dmlist)]=0

      def extract_metadata_galactic(self):
# Extracting Galactic Data
          flaglist=numpy.where(self.photoz_list==0.0,1,0)
          [self.ra]=numpy.compress(flaglist,self.ra_list)
          [self.decl]=numpy.compress(flaglist,self.decl_list)
          [self.gall]=numpy.compress(flaglist,self.gall_list)
          [self.galb]=numpy.compress(flaglist,self.galb_list)
          [self.ddf_bool]=numpy.compress(flaglist,self.ddf_list)
          [self.host_specz]=numpy.compress(flaglist,self.specz_list)
          [self.host_photoz]=numpy.compress(flaglist,self.photoz_list)
          [self.host_photozerr]=numpy.compress(flaglist,self.photozerr_list)
          [self.distmod]=numpy.compress(flaglist,self.dm_list)
          [self.mwebv]=numpy.compress(flaglist,self.mwebv_list)
          [self.target]=numpy.compress(flaglist,self.target_list)

      def writeout_trainingltcv(self,objectid):
          self.extract_ltcv(objectid)
          objname=str(objectid)
          if(self.ddf_bool==1): ddfdir='ddf'
          if(self.ddf_bool==0): ddfdir='wide'
          classdir='../ltcv_training/'+ddfdir+'/class_'+str(self.target)
          if not (os.path.exists(classdir)): os.mkdir(classdir)
          asciifiledir='../ltcv_training/'+ddfdir+'/class_'+str(self.target)+'/'+objname
          if not (os.path.exists(asciifiledir)): os.mkdir(asciifiledir)
          asciifilename='../ltcv_training/'+ddfdir+'/class_'+str(self.target)+'/'+objname+'/'+objname+'.dat'
          print(asciifilename)
          outputfile=open(asciifilename,'w')
          outputfile.write("time band flux fluxerr zp zpsys flag"+"\n") 
          for j in range(len(self.mjdlist)):
             outputfile.write("%12.3f"%(self.mjdlist[j])+\
             "   "+"%4i"%(self.filterlist[j])+\
             "   "+"%12.3f"%(self.fluxlist[j])+\
             "   "+"%12.3f"%(self.fluxerrlist[j])+\
             "   "+"%8.1f"%(27.5)+\
             "   "+"%5s"%("ab")+\
             "   "+"%4i"%(self.flaglist[j])+"\n")
          outputfile.close()

      def writeout_ltcv(self,objectid):
          self.extract_ltcv(objectid)
          objname=str(objectid)
          if(self.ddf_bool==1): ddfdir='ddf'
          if(self.ddf_bool==0): ddfdir='wide'
          if(self.photoz>0.0):  galacticdir='../ltcv/'+ddfdir+'/extragalactic/'
          if(self.photoz==0.0): galacticdir='../ltcv/'+ddfdir+'/galactic/'
          if not (os.path.exists(galacticdir)): os.mkdir(galacticdir)
          asciifiledir=galacticdir+'/'+objname
          if not (os.path.exists(asciifiledir)): os.mkdir(asciifiledir)
          asciifilename=asciifiledir+'/'+objname+'.dat'
          print(asciifilename)
          outputfile=open(asciifilename,'w')
          outputfile.write("time band flux fluxerr zp zpsys flag"+"\n")
          for j in range(len(self.mjdlist)):
             outputfile.write("%12.3f"%(self.mjdlist[j])+\
             "   "+"%4i"%(self.filterlist[j])+\
             "   "+"%12.3f"%(self.fluxlist[j])+\
             "   "+"%12.3f"%(self.fluxerrlist[j])+\
             "   "+"%8.1f"%(27.5)+\
             "   "+"%5s"%("ab")+\
             "   "+"%4i"%(self.flaglist[j])+"\n")
          outputfile.close()

      def extract_ltcvmax(self,objectid):
          self.maxflux=numpy.zeros(6)
          for i in range(6):
             filterflaglist=numpy.where(self.filterlist==i,1,0)
             self.maxfluxlist=numpy.compress(filterflaglist,self.fluxlist)
             self.maxflux[i]=numpy.max(self.maxfluxlist)
             del self.maxfluxlist ; del filterflaglist
                 
      def write_ltcv(self):
          if(self.ddf==True):
            objectdir=ddfdir+'/'+'object_'+str(self.target)+'/id'+"%04i"%(self.object_id)
            if(os.path.exists(objectdir)==False):
               os.mkdir(objectdir)
          if(self.ddf==False):
            objectdir=widedir+'/'+'object_'+str(self.target)+'/id'+"%04i"%(self.object_id)
            if(os.path.exists(objectdir)==False):
               os.mkdir(objectdir)

      #def write_ltcv(objectid):
      def target_class(self):
          targetclass=[6,15,16,42,52,53,62,64,65,67,88,90,92,95]

      def read_trainingfluxlist(self,classnumber):
          fluxfile=open('../data_processed/training_maxflux.dat','r')
          fluxdata=fluxfile.readlines()
          fdata=numpy.genfromtxt(fluxdata,dtype="i4,a10,i4,f4,f4,f4,f4,f4,f4,f4,f4,f4",\
          names=['objclassnumber','objclass','objid','specz','photoz','dm',\
                 'u','g','r','i','z','y'],\
          comments='#',skip_header=1)

          self.objclassnumber=fdata[:]['objclassnumber']
          self.objclass=fdata[:]['objclass']
          self.objid=fdata[:]['objid']
          self.specz=fdata[:]['specz']
          self.photoz=fdata[:]['photoz']
          self.u=fdata[:]['u']
          self.g=fdata[:]['g']
          self.r=fdata[:]['r']
          self.i=fdata[:]['i']
          self.z=fdata[:]['z']
          self.y=fdata[:]['y']

          if(classnumber!=0):
            classflaglist=numpy.where(self.objclassnumber==classnumber,1,0)
            #print(classflaglist)
            #classflaglist=numpy.where(self.objclass=='class_'+str(classnumber),1,0)
            self.objid=numpy.compress(classflaglist,self.objid)
            self.specz=numpy.compress(classflaglist,self.specz)
            self.photoz=numpy.compress(classflaglist,self.photoz)
            self.u=numpy.compress(classflaglist,self.u)
            self.g=numpy.compress(classflaglist,self.g)
            self.r=numpy.compress(classflaglist,self.r)
            self.i=numpy.compress(classflaglist,self.i)
            self.z=numpy.compress(classflaglist,self.z)
            self.y=numpy.compress(classflaglist,self.y)

def load_lsstdictionary():
          lsstcolor={0:'#7b68ee',1:'#0000ff',2:'#008000',3:'#ffa500',4:'red',5:'#800080'}
          lsstfilter={0:'u',1:'g',2:'r',3:'i',4:'z',5:'y'}
          #plotstyle={1:'bo',2:'go',3:'ro',4:'co',5:'mo'}
          return [lsstcolor, lsstfilter]

