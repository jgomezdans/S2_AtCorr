#!/usr/bin/env python

"""
Extract S2 L1, S2 L2, MODIS and Landsat data to do comparisons


"""
import datetime
import glob
import os
import sys
from collections import namedtuple

from sklearn import linear_model

import numpy as np
import gdal
import pylab as plt

from helper_functions import reproject_image_to_master, hplot
from helper_functions import parse_xml

parent_folder = "/data/selene/ucfajlg/S2_AC/TeleSpazio/" + \
                "ftp.telespazio.fr/outgoing/L2A_Products/"

MODIS_DIR = "/data/selene/ucfajlg/S2_AC/MCD43/"

BOA_list = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12", "AOT",
            "WVP", "SCL_20", "SCL_60"]
TOA_list = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12"]

BOA_set = namedtuple("BOA", 
                "b1 b2 b3 b4 b5 b6 b7 b8 " + 
                "b8a b9 b10 b11 b12 aot wv scl_20 scl_60")
TOA_set = namedtuple("BOA", 
                "b1 b2 b3 b4 b5 b6 b7 b8 " + 
                "b8a b9 b10 b11 b12")


class TeleSpazioComparison(object):

    def __init__ (self, site, tile):
        """
        site is e.g. 'Ispra'. It is used to find folders. 'tile'
        is just the UTM tile being used for that particular 
        site.
        """
        self.site = site
        self.tile = tile
        self.l1c_files = self._get_safe_files("L1C")
        self.l2a_files = self._get_safe_files("L2A")
        self.l1c_datasets = {}
        self.l2a_datasets = {}
        for the_date in self.l1c_files.iterkeys():
            retval = self.get_l1c_data(the_date)
            if retval is None:
                continue
                # No tile found
            self.l1c_datasets[the_date]=retval
            self.l2a_datasets[the_date] = self.get_l2_data(
                the_date)
                
        
        
    def _get_safe_files(self, file_type):
        """A method that looks for SAFE files and
        returns a dictionary ordered by date. You can select
        whether you want the L1C or L2C files.
        """
        
        if file_type.upper() == "L1C":
            files = glob.glob(os.path.join(parent_folder, 
                "{}/L1C/*.SAFE".format(self.site)))
        elif file_type.upper() == "L2A":
            files = glob.glob(os.path.join(parent_folder, 
                "{}/zip/*.SAFE".format(self.site)))
        else:
            raise ValueError("Can only deal with L1C and L2A")
        files.sort()
        time_strx = [os.path.basename(fich).split("_")[7]
                        for fich in files]
        time = [datetime.datetime.strptime(time_str,
                "V%Y%m%dT%H%M%S") for time_str in time_strx]
        return dict(zip(time,files))
        

    def get_l1c_angles(self, the_date):
        """Gets the mean view/illumination angles from the L1C 
        metadata. Returns SZA, SAA, VZA and VAA in degrees.
        """
        l1c_dir = self.l1c_files[the_date]
        granule_dir0 = os.path.join(l1c_dir, "GRANULE/")
        for granule in os.listdir(granule_dir0):
            print granule
            if granule.find(self.tile) >= 0:
                granule_dir = os.path.join(granule_dir0,
                                           granule)
        try:
            xml_file = [f for f in os.listdir(granule_dir) 
                        if f.find(".xml") >= 0][0]
        except:
            return None
        angles = parse_xml(os.path.join(granule_dir,xml_file))
        return angles
    
    def get_l1c_data(self, the_date):
        """
        Grabs the L1C data. It's getting it from the TeleSpazio
        part of the archive, not from my other downloads. They
        should be the same though...
        Returns a TOA_set object, with the bands at their
        native resolutions
        """
        l1c_dir = self.l1c_files[the_date]
        granule_dir0 = os.path.join(l1c_dir, "GRANULE/")
        for granule in os.listdir(granule_dir0):
            if granule.find(self.tile) >= 0:
                granule_dir = os.path.join(granule_dir0,
                                           granule)
        try:
            files = glob.glob(os.path.join(granule_dir, 
                                       "IMG_DATA", "*.jp2"))
        except:
            return None
        files.sort()
        data = []
        for band in TOA_list:
            for fich in files:
                if fich.find(band) >= 0:
                    data.append(fich)
        datasets = TOA_set(*data)
        return datasets

    def get_l2_data(self, the_date):
        """
        Gets hold of the TeleSpazio Level 2A data. It requires
        a datetime key that includes the acquisition of the
        data. It returns a BOA_set with all the bands at their
        native resolution, WV, AOT and the 20 and 60 m scene
        masks.
        """
        l2a_dir = self.l2a_files[the_date]
        granule_dir0 = os.path.join(l2a_dir, "GRANULE/")
        for granule in os.listdir(granule_dir0):
            if granule.find(self.tile) >= 0:
                granule_dir = os.path.join(granule_dir0,
                                           granule)
        try:
            files = glob.glob(os.path.join(granule_dir,
                                           "IMG_DATA",
                                           "*SCL*.jp2"))
        except:
            return None
        files.sort()
        selected_band = {}
        scl_20m = files[0]
        scl_60m = files[1]
        study_bands = {'SCL_20': files[0], 
                       'SCL_60': files[1]}
        for resolution in ["R10m", "R20m", "R60m"]:
            
            files = glob.glob(os.path.join(granule_dir,
                                           "IMG_DATA",
                                           resolution,
                                           "*.jp2"))
            
            if resolution == "R10m":
                for ii,selected_band in enumerate([ 
                    "B02", "B03", "B04", "B08"]):
                    for fich in files:
                        if fich.find("{}_10m.jp2".format(
                            selected_band)) >= 0:
                            study_bands[selected_band] = fich
            elif resolution == "R20m":
                for ii,selected_band in enumerate([ 
                    "B05", "B06", "B07", "B11", "B12", "B8A"]):
                    for fich in files:
                        if fich.find("{}_20m.jp2".format(
                            selected_band)) >= 0:
                            study_bands[selected_band] = fich
                            print fich
                for ii,selected_band in enumerate(["AOT", 
                                                   "WVP"]):
                    for fich in files:
                        if (fich.find("{}_".format(
                            selected_band)) >= 0) and \
                            (fich.find(".jp2") >= 0):
                            study_bands[selected_band] = fich
                            print fich

            elif resolution == "R60m":
                for ii,selected_band in enumerate([ 
                    "B01", "B09"]):
                    for fich in files:
                        if fich.find("{}_60m.jp2".format(
                            selected_band)) >= 0:
                            study_bands[selected_band] = fich
                study_bands["B10"] = None
        dataset = [ study_bands[k] for k in BOA_list]
        return BOA_set(*dataset)
        
    def do_scatter(self, the_date, band, sub=10):
        toa_set = self.get_l1c_data(the_date)
        boa_set = self.get_l2_data(the_date)
        g = gdal.Open(toa_set[TOA_list.index(band)])
        toa_rho = g.ReadAsArray()
        g = gdal.Open(boa_set[BOA_list.index(band)])
        boa_rho = g.ReadAsArray()
        mask_toa = np.logical_or(toa_rho == 0,
                                  toa_rho > 20000)
        mask_boa = np.logical_or(boa_rho == 0,
                                  boa_rho > 20000) 
        toa_rho = toa_rho/10000.
        boa_rho = boa_rho/10000.
        mask = mask_boa*mask_toa
        import pdb;pdb.set_trace()
        hplot(boa_rho[~mask][::sub], toa_rho[~mask][::sub])

    def get_transform(self, the_date, band, 
                      sub=10, nv=200, lw=2, odir='figures',
                      apply_model=False):

        # ensure odir exists
        if not os.path.exists(odir): os.makedirs(odir)

        fname = odir+'/'+'%s_%s'%(the_date.strftime("%Y-%m-%d %H:%M:%S"), band)

        toa_set = self.get_l1c_data(the_date)
        boa_set = self.get_l2_data(the_date)
        if toa_set is None or boa_set is None:
            print "No TILEs found for %s" % the_date
            return None
        g = gdal.Open(toa_set[TOA_list.index(band)])
        toa_rho = g.ReadAsArray()
        g = gdal.Open(boa_set[BOA_list.index(band)])
        boa_rho = g.ReadAsArray()
        mask_toa = np.logical_or(toa_rho == 0,
                                  toa_rho > 20000)
        mask_boa = np.logical_or(boa_rho == 0,
                                  boa_rho > 20000)
        toa_rho = toa_rho/10000.
        boa_rho = boa_rho/10000.
        mask = mask_boa*mask_toa
        x = boa_rho[~mask][::sub]
        y = toa_rho[~mask][::sub]

        vmin = np.min([0.0,np.min(x),np.min(y)])
        vmax = np.max([1.0,np.max(x),np.max(y)])

        ns = x.size
        # robust linear model fit
        model = linear_model.LinearRegression(n_jobs=-1)
        model_ransac = linear_model.RANSACRegressor(model)
        model_ransac.fit(y.reshape(ns,1), x) 
        inlier_mask = model_ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_X = np.arange(vmin,vmax,(vmax-vmin)/nv)       
        line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])
        xlim = ylim = [vmin,vmax]

        hplot(x, y, new=True,xlim=xlim,ylim=ylim)
        xyrange = xlim,ylim
        plt.xlim(xlim)
        plt.plot(xyrange[0],xyrange[1],'g--', label='1:1 line')
        plt.plot(line_y_ransac, line_X, color='red', linestyle='-',linewidth=lw, label='RANSAC regressor') 
        plt.xlabel('BOA reflectance Band %s'%band)
        plt.ylabel('TOA reflectance Band %s'%band)

        a,b = model_ransac.predict(np.array([0.,1.])[:, np.newaxis])
        plt.title(the_date.strftime("%Y-%m-%d %H:%M:%S") + \
		'\nBOA(%s) = %.3f + %.3f TOA(%s)'%(band,a,b-a,band) + \
		'\nTOA(%s) = %.3f + %.3f BOA(%s)'%(band,a/(a-b),1./(b-a),band))

        if vmax > 1:
            plt.plot(xlim,[1.0,1.0],'k--',label='TOA reflectance == 1')
        plt.legend(loc='best')
        plt.savefig(fname+'.scatter.pdf')
        plt.close() 
        if apply_model:
            approx_boa_rho = model_ransac.predict(toa_rho[~mask].flatten()[:, 
                                                                 np.newaxis])
            retval = np.zeros_like (toa_rho)
            retval[~mask] = approx_boa_rho
        return model_ransac, retval
        

    def get_modis_files(self, site):
        """Gets the MODIS files. You get in return a dictionary
        with the same keys as the L1 andHCD43A2 products.
        """
        layer_selector = {
            'MCD43A1':['HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band'+"%d"%(b+1) for b in xrange(7)],
            'MCD43A2':['HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Uncertainty','HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:Snow_BRDF_Albedo']}
        modis_mapper = {}
        for product in [ "MCD43A1", "MCD43A2"]:
            modis_mapper[product] = {}
            files = glob.glob(
                os.path.join(MODIS_DIR, site) + 
                "/%s*.hdf" % product)
            files.sort()
            modis_dates_t = [f.split("/")[-1].split(".")[1][1:] 
                                for f in files]
            modis_dates = [datetime.datetime.strptime(d, 
                        "%Y%j")
                        for d in modis_dates_t]
            
            for s2_date in self.l2a_datasets.iterkeys():
                for i, modis_date in enumerate(modis_dates):
                    if modis_date.date() == s2_date.date():
                        master=self.get_l2_data(s2_date).scl_20
                        # Now, reproject/crop all layers for
                        # this product...
                        fnames = []
                        for layer in layer_selector[product]:
                            fnames.append(
                                reproject_image_to_master (
                                master, files[i], layer))

                        modis_mapper[product][s2_date] = fnames
        return modis_mapper


if __name__ == "__main__":
    ts = TeleSpazioComparison("Ispra", "T32TMR")
    for ii, the_date in enumerate( ts.l1c_files.iterkeys()):
        print ts.get_l1c_data(the_date)
        if ii == 5:
            break

    # do scatter plot and get transform
    model, boa_approx = ts.get_transform(the_date, "B02", apply_model=True)
    #modis_times = ts.get_modis_files("Ispra")
        
