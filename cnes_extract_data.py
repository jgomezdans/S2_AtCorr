#!/usr/bin/env python

"""
Extract S2 L1, S2 L2, MODIS and Landsat data to do comparisons


"""
import datetime
import glob
import os
import sys
import zipfile
from collections import namedtuple


import numpy as np
from sklearn import linear_model
import scipy.ndimage
import gdal
import pylab as plt

from helper_functions import reproject_image_to_master, hplot
from helper_functions import parse_xml, load_emulator_training_set

from extract_data import TeleSpazioComparison


CNES_DIR = "/data/selene/ucfajlg/S2_AC/CNES/V1_1/"
MODIS_DIR = "/data/selene/ucfajlg/S2_AC/MCD43/"

TOA_list = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12"]

CNES_BOA_set = namedtuple("CNES_BOA", 
                          "b2 b3 b4 b5 b6 b7 b8 " + 
                          "b8a b11 b12 atb_r1 atb_r2 " +
                          "sat_r1 sat_r2 clm_r1 clm_r2 mg2_r1 mg2_r2")
CNES_BOA_list = [ "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B11", "B12"]

# B1, B8 and B10 are not in CNES dataset



class CNESComparison(TeleSpazioComparison):
    """A Class to do comparisons of the CNES L2A product."""
    def __init__ (self, site, tile):
        # The parent class creator uses the TeleSpazio data to fetch the L1C
        # products

        TeleSpazioComparison.__init__(self, site, tile)
        self.l2a_datasets = {}
        self.l2a_files = None
        self.__find_l2a_data()
        
        
        # self.l1c_files is now defined, and starts to look for L2A products
        
    ###def __find_l2a_data(self):
        ###self.l2a_files = self._get_safe_files("L2A")
        ###self.l1c_datasets = {}
        ###self.l2a_datasets = {}
        ###for the_date in self.l1c_files.iterkeys():
            ###retval = self.get_l1c_data(the_date)
            ###if retval is None:
                ###continue
                #### No tile found
            ###self.l1c_datasets[the_date]=retval
            ###self.l2a_datasets[the_date] = self.get_l2_data(
                ###the_date)
    def __find_l2a_data(self):         
        granules = glob.glob( os.path.join(CNES_DIR, "SENTINEL2A*"))
        granules.sort()
        self.l2a_files = {}
        for granule in granules:
            if granule.find(self.tile) >= 0: # We have the tile!
                
                tstring = granule.split("/")[-1].split("_")[1]
                key = datetime.datetime.strptime( tstring, 
                                "%Y%m%d-%H%M%S-%f").replace(second=0,
                                                            microsecond=0)
                print "Dealing with iamge acquired on %s" % key
                
                x = self._unpack_data(granule)
                self.l2a_files[key] = x
            
        
        for k in self.l2a_files.iterkeys():
            try:
                r = self.l1c_datasets[k]
                self.l2a_datasets[k] = self.l2a_files[k]
            except KeyError:
                continue
            
            
            
       
    def _unpack_data(self, granule, product=None):
        # CNES data are zipped up. This unzips the files up and returns a list 
        # of files back
        
        zipname = os.path.join(granule, granule.split("/")[-1]+".zip")
        zipper = zipfile.ZipFile(zipname)
        
        if not os.path.exists(os.path.join(granule, "MASKS")):
            # Uncompress data
            print "Uncompressing zipfile"
            zipper.extractall(CNES_DIR)
        files = []
        tags = []
        for product in ["SRE", "ATB", "SAT", "CLM", "MG2"]:
            for fich in zipper.namelist():
                if fich.find(product) >= 0:
                    fname = fich.split("/")[-1]
                    tag = "_".join(fname.replace(".tif","").split("_")[-2:]).lower()
                    if product == "SRE":
                        tag = tag.replace("sre_", "")
                    tags.append(tag)
                    files.append ( os.path.join(CNES_DIR, fich))
        zipper.close()
        
        files2 = []
        tago = ('b2 b3 b4 b5 b6 b7 b8 b8a b11 b12 atb_r1 ' + 
            'atb_r2 sat_r1 sat_r2 clm_r1 clm_r2 mg2_r1 mg2_r2').split()
        for t in tago:
                files2.append(files[tags.index(t)])
        files = CNES_BOA_set(*files2)
        return files
        
    def get_transform(self, the_date, band, mask="L2",
                      sub=10, nv=200, lw=2, odir='figures',
                      apply_model=True, plausible=True):

        # ensure odir exists
        if not os.path.exists(odir): os.makedirs(odir)

        fname = odir+'/'+'MAJA_%s_%s_%s_%s'%(
            self.site, self.tile, the_date.strftime("%Y-%m-%d %H:%M:%S"), band)


        toa_set = self.get_l1c_data(the_date)
        boa_set = self.l2a_datasets[the_date]
        if toa_set is None or boa_set is None:
            print "No TILEs found for %s" % the_date
            return None
        g = gdal.Open(toa_set[TOA_list.index(band)])
        toa_rho = g.ReadAsArray()
        g = gdal.Open(boa_set[CNES_BOA_list.index(band)])
        boa_rho = g.ReadAsArray()
        if mask == "L2":
            # TODO NEEDS WORK -> use clm_r1/r2 filter veg?
            # if mg2, times by 01000000 if 0 no cloud 
            print "Using L2A product mask"
            if band in ["B02", "B03", "B04", "B08"]:
                g = gdal.Open(boa_set.mg2_r1)
                c = g.ReadAsArray()
                mask = np.bitwise_and(2, c) == 2
                
            elif band in ["B05", "B06", "B07", "B11", "B12", "B8A"]:
                g = gdal.Open(boa_set.mg2_r2)
                c = g.ReadAsArray()
                mask = np.bitwise_and(2, c) == 2
                

        else:
            mask_toa = np.logical_or(toa_rho == 0,
                                    toa_rho > 20000)
            mask_boa = np.logical_or(boa_rho == 0,
                                    boa_rho > 20000)
            mask = mask_boa*mask_toa

        toa_rho = toa_rho/10000.
        boa_rho = boa_rho/10000.
        
        x = boa_rho[~mask][::sub]
        y = toa_rho[~mask][::sub]
        
        vmin = np.min([0.0,np.min(x),np.min(y)])
        vmax = np.max([1.0,np.max(x),np.max(y)])
        line_X = np.arange(vmin,vmax,(vmax-vmin)/nv)               
        ns = x.size
        xlim = ylim = [vmin,vmax]
        # robust linear model fit
        model = linear_model.LinearRegression(n_jobs=-1)
        hplot(x, y, new=True,xlim=xlim,ylim=ylim)
        xyrange = xlim,ylim
        plt.xlim(xlim)
        plt.plot(xyrange[0],xyrange[1],'g--', label='1:1 line')

        try:
            model_ransac = linear_model.RANSACRegressor(model)
            model_ransac.fit(y.reshape(ns,1), x) 
            inlier_mask = model_ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])
            plt.plot(line_y_ransac, line_X, color='red', linestyle='-',
                     linewidth=lw, label='RANSAC regressor') 
            a,b = model_ransac.predict(np.array([0.,1.])[:, np.newaxis])
            plt.title(the_date.strftime("%Y-%m-%d %H:%M:%S") + \
                '\nBOA(%s) = %.3f + %.3f TOA(%s)'%(band,a,b-a,band) + \
                '\nTOA(%s) = %.3f + %.3f BOA(%s)'%(band,a/(a-b),1./(b-a),band))
            if apply_model:
                approx_boa_rho = model_ransac.predict(toa_rho[~mask].flatten()[:, 
                                                                 np.newaxis])
                retval = np.zeros_like (toa_rho)
                retval[~mask] = approx_boa_rho


        except ValueError:
            model_ransac = None
            retval = None

        
        plt.xlabel('BOA reflectance Band %s'%band)
        plt.ylabel('TOA reflectance Band %s'%band)


        if vmax > 1:
            plt.plot(xlim,[1.0,1.0],'k--',label='TOA reflectance == 1')
        if plausible:
            boa_emu, toa_emu = load_emulator_training_set()
            plt.plot(boa_emu, toa_emu[band], '+', markersize=3, c="cyan", label="Plausible")
        plt.legend(loc='best')
        plt.savefig(fname+'.scatter.pdf')
        plt.close() 
        return model_ransac, retval

                
        

if __name__ == "__main__":

    refl_list = [ "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A",  "B11", "B12"]
    
    for (site,tile) in [ ["Ispra", "T32TMR"], 
                        ["Pretoria", "35JPM"], ["Pretoria", "35JQM"]]:
        ts = CNESComparison(site, tile)
        for the_date in ts.l2a_datasets.iterkeys():
            for band in refl_list:
                ts.get_transform(the_date, band)    
