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

import re

# select bands...
regex = r"B[0-9][0-9,A]"

##GFZ_DIR = "/data/selene/ucfajlg/S2_AC/GFZ/doidata.gfz-potsdam.de/S2_AC_FS/S2A_MSI_L2A/v0.10/"
GFZ_DIR = "/storage/ucfajlg/S2_AC/GFZ/doidata.gfz-potsdam.de/S2_AC_FS/S2A_MSI_L2A/v0.10/"
MODIS_DIR = "/data/selene/ucfajlg/S2_AC/MCD43/"

TOA_list = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09",  "B11", "B12"]

GFZ_BOA_set = namedtuple("GFZ_BOA", 
                          "b1 b2 b3 b4 b5 b6 b7 b8 " + 
                          "b8a b11 b12 aot wvc " + 
                          "msk_10m msk_20m msk_60m " + 
                          "pml2a_10m pml2a_20m pml2a_60m")

GFZ_BOA_list = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B11", "B12", "AOT", "WVC",
            "MSK_10m", "MSK_20m", "MSK_60m", "PML2A_10m",
            "PML2A_20m", "PML2A_60m"]

# B1, B8 and B10 are not in CNES dataset



class GFZComparison(TeleSpazioComparison):
    """A Class to do comparisons of the CNES L2A product."""
    def __init__ (self, site, tile):
        # The parent class creator uses the TeleSpazio data to fetch the L1C
        # products

        TeleSpazioComparison.__init__(self, site, tile)
        self.l2a_datasets = {}
        self.l2a_files = None
        self.__find_l2a_data()
        
        
        # self.l1c_files is now defined, and starts to look for L2A products
        

    def __find_l2a_data(self):         
        #tile is T32TMR (eg)
        # this goes in /data/selene/ucfajlg/S2_AC/GFZ/doidata.gfz-potsdam.de/S2_AC_FS/S2A_MSI_L2A/v0.10/32T/MR/2016/
        self.l2a_files = {}
        for cur, _dirs, files in os.walk(GFZ_DIR):
            if cur.find(".SAFE") >=1 and cur.endswith("GRANULE"):
                try:
                    
                    tstring = cur.split("/")[-2].split("_")[7][1:]
                    tile = _dirs[0].split("_")[9]
                except ValueError:
                    continue # dodgy file
                if tile == self.tile:
                    
                    key = datetime.datetime.strptime( tstring, 
                                "%Y%m%dT%H%M%S").replace(second=0,
                                                            microsecond=0)
                    print "Dealing with iamge acquired on %s" % key
                
                    self.l2a_files[key] = os.path.join(cur, _dirs[0])
                    
            
        
        for k in self.l2a_files.iterkeys():
            try:
                r = self.l1c_datasets[k]
                self.l2a_datasets[k] = self._gfz_sorter( self.l2a_files[k])
                print k
            except KeyError:
                continue
    
    def _gfz_sorter(self, path):
        granule = {}
        for cur, _dir, files in os.walk(path):
            
            for fich in files:
                
                if fich.endswith(".jp2"):
                    # Need to spelunk this fname for
                    # the tiff file...
                    suffix = cur.split("/")[-1]
                    prefix = "/".join(cur.split("/")[:-1])
                    fname = suffix + fich.replace("jp2", "tif")
                    reso = fich.split(".")[0].split("_")[-1] 
                    # should be 10m, 20m, 60m
                    if fich.find("AOT") >= 0:
                        granule["AOT"] = os.path.join(prefix, fname)
                    elif fich.find("CWV") >= 0:
                        granule["WVC"] =os.path.join(prefix, fname)
                    elif fich.find("MSK") >= 0:
                        granule["MSK_%s" % reso] =  os.path.join(prefix, fname)
                    elif fich.find("PML2A") >= 0:
                        granule["PML2A_%s" % reso] = os.path.join(prefix, fname)
                    elif re.search(regex, fich):
                        band_no = fich.split("_")[-2] # (eg B02)
                        granule[band_no] = os.path.join(prefix, fname)
        
        data = []
        for d in GFZ_BOA_list:
            if not os.path.exists ( granule[d]):
                raise IOError
            data.append(granule[d])
            
        return GFZ_BOA_set(*data)
        
    def get_transform(self, the_date, band, mask="L2",
                      sub=10, nv=200, lw=2, odir='figures',
                      apply_model=True, plausible=True):

        # ensure odir exists
        if not os.path.exists(odir): os.makedirs(odir)

        fname = odir+'/'+'GFZ_%s_%s_%s_%s'%(
            self.site, self.tile, the_date.strftime("%Y-%m-%d %H:%M:%S"), band)

        print fname
        toa_set = self.get_l1c_data(the_date)
        boa_set = self.l2a_datasets[the_date]
        if toa_set is None or boa_set is None:
            print "No TILEs found for %s" % the_date
            return None
        print toa_set[TOA_list.index(band)]
        print boa_set[GFZ_BOA_list.index(band)]
        g = gdal.Open(toa_set[TOA_list.index(band)])
        toa_rho = g.ReadAsArray()
        g = gdal.Open(boa_set[GFZ_BOA_list.index(band)])
        boa_rho = g.ReadAsArray()
        if mask == "L2":
            # reelvant MSK set to 10 (clear)
            #relevant PMSL set to 1
            print "Using L2A product mask"
            if band in ["B02", "B03", "B04", "B08"]:
                g = gdal.Open(boa_set.msk_10m)
                c1 = g.ReadAsArray()
                g = gdal.Open(boa_set.pml2a_10m)
                c2 = g.ReadAsArray()
                #mask = np.logical_and( c1==10, c2 == 1)
                mask = c1==10
                
            elif band in ["B05", "B06", "B07", "B11", "B12", "B8A"]:
                g = gdal.Open(boa_set.msk_20m)
                c1 = g.ReadAsArray()
                g = gdal.Open(boa_set.pml2a_20m)
                c2 = g.ReadAsArray()
                mask = c1==10
                print c1.shape, c2.shape
                #mask = np.logical_and( c1==10, c2 == 1)
            elif band in ["B01"] : # 60m
                g = gdal.Open(boa_set.msk_60m)
                c1 = g.ReadAsArray()
                g = gdal.Open(boa_set.pml2a_60m)
                c2 = g.ReadAsArray()
                mask = c1==10
                #mask = np.logical_and( c1==10, c2 == 1)


        else:
            mask_toa = np.logical_or(toa_rho == 0,
                                    toa_rho > 20000)
            mask_boa = np.logical_or(boa_rho == 0,
                                    boa_rho > 20000)
            mask = mask_boa*mask_toa
        mask_boa = np.logical_or(boa_rho == 0,
                                boa_rho > 20000)
        mask_toa = np.logical_or(toa_rho == 0,
                                    toa_rho > 20000)
        mask = np.logical_and( c1==10, ~mask_boa)
        
        toa_rho = toa_rho/10000.
        boa_rho = boa_rho/10000.
        
        x = boa_rho[mask][::sub]
        y = toa_rho[mask][::sub]
        print "Masked data"
        vmin = np.min([0.0,np.min(x),np.min(y)])
        vmax = np.max([1.0,np.max(x),np.max(y)])
        print vmin, vmax
        line_X = np.arange(vmin,vmax,(vmax-vmin)/nv)               
        ns = x.size
        xlim = ylim = [vmin,vmax]
        # robust linear model fit
        print "Fitting linear model"
        model = linear_model.LinearRegression(n_jobs=-1)
        hplot(x, y, new=True,xlim=xlim,ylim=ylim)
        xyrange = xlim,ylim
        plt.xlim(xlim)
        plt.plot(xyrange[0],xyrange[1],'g--', label='1:1 line')
        retval = None
        try:
            print "Fitting RANSAC model"
            model_ransac = linear_model.RANSACRegressor(model)
            model_ransac.fit(y.reshape(ns,1), x) 
            inlier_mask = model_ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            print "RANSAC model predictions"
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
            print "RANSAC failed"
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
        print "Saved fname"
        return model_ransac, retval

                
        

if __name__ == "__main__":

            
    #for (site,tile) in [["Pretoria", "35JPM"], ["Ispra", "T32TMR"], 
                        #["Pretoria", "35JQM"]]:
    for (site,tile) in [ ["Ispra", "T32TMR"]]:

        ts = GFZComparison(site, tile)
        
        for ii, the_date in enumerate( ts.l2a_datasets.iterkeys()):        
            print the_date
            ts.compare_boa_refl_MCD43(the_date, 1)
