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
import kernels
from spatial_mapping import *

parent_folder = "/data/selene/ucfajlg/S2_AC/TeleSpazio/" + \
                "ftp.telespazio.fr/outgoing/L2A_Products/"

CNES_DIR = "/data/selene/ucfajlg/S2_AC/CNES/V1_1/"
MODIS_DIR = "/data/selene/ucfajlg/S2_AC/MCD43/"

BOA_list = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12", "AOT",
            "WVP", "SCL_20", "SCL_60", "CLD_20", "CLD_60", 
            "SNW_20", "SNW_60"]
TOA_list = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12"]

BOA_set = namedtuple("BOA", 
                "b1 b2 b3 b4 b5 b6 b7 b8 " + 
                "b8a b9 b10 b11 b12 aot wv scl_20 scl_60 " + 
                "cld_20 cld_60 snw_20 snw_60")
TOA_set = namedtuple("BOA", 
                "b1 b2 b3 b4 b5 b6 b7 b8 " + 
                "b8a b9 b10 b11 b12")

CNES_BOA_set = namedtuple("CNES_BOA", 
                          "b2 b3 b4 b5 b6 b7 b8 " + 
                          "b8a b11 b12 atb_r1 atb_r2 " +
                          "sat_r1 sat_r2 clm_r1 clm_r2 mg2_r1 mg2_r2")
CNES_BOA_list = [ "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B11", "B12"]

# B1, B8 and B10 are not in CNES dataset

class TeleSpazioComparison(object):
    """A class for analysing the TeleSpazio AtCorr data
    """
    def __init__ (self, site, tile):
        """
        site is e.g. 'Ispra'. It is used to find folders. 'tile'
        is just the UTM tile being used for that particular 
        site.
        """
        self.site = site
        self.tile = tile
        self.l1c_files = self._get_safe_files("L1C")
        self.__find_l2a_data()
        self.modis_files(tile, site)
        
    def __find_l2a_data(self):
        self.l2a_files = self._get_safe_files("L2A")
        self.l1c_datasets = {}
        self.l2a_datasets = {}
        for the_date in self.l1c_files.iterkeys():
            retval1 = self.get_l1c_data(the_date)
            retval2 = self.get_l2_data(the_date)
            if retval1 is None or retval2 is None:
                continue
                # No tile found            
            else:
                self.l1c_datasets[the_date] = retval1    
                self.l2a_datasets[the_date] = retval2

        
    def _get_mask(self, the_date, res="20"):
        # Get the data
        boa_set = self.get_l2_data(the_date)
        if res == "20":
            g = gdal.Open(boa_set.scl_20)
            class_id = g.ReadAsArray()
        elif res == "60":
            g = gdal.Open(boa_set.scl_60)
            class_id = g.ReadAsArray()
        elif res == "10":
            g = gdal.Open(boa_set.scl_20)
            class_id20 = g.ReadAsArray()
            class_id = scipy.ndimage.zoom( class_id20, 2, order=0)
            
        # Select water (6), bare soils (5) and vegetation (4)
        # Ignoring class 11 (snow/ice). All others are crap pixels
        mask = np.logical_and(class_id >= 4, class_id <= 6)

        return mask
            

        
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
        times = [t.replace(second=0) for t in time]
        return dict(zip(times,files))
        

    def get_l1c_angles(self, the_date):
        """Gets the mean view/illumination angles from the L1C 
        metadata. Returns SZA, SAA, VZA and VAA in degrees.
        """
        l1c_dir = self.l1c_files[the_date]
        granule_dir0 = os.path.join(l1c_dir, "GRANULE/")
        for granule in os.listdir(granule_dir0):
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

        the_date = the_date.replace(second=0)
        l1c_dir = None
        for kdate in self.l1c_files.iterkeys():
            k = kdate.replace(second=0)
            if the_date == k:
                l1c_dir = self.l1c_files[kdate]
        if l1c_dir is None:
            print "No pairing for %s" % the_date
            return None
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

                for ii,selected_band in enumerate(["AOT", 
                                                   "WVP"]):
                    for fich in files:
                        if (fich.find("{}_".format(
                            selected_band)) >= 0) and \
                            (fich.find(".jp2") >= 0):
                            study_bands[selected_band] = fich


            elif resolution == "R60m":
                for ii,selected_band in enumerate([ 
                    "B01", "B09"]):
                    for fich in files:
                        if fich.find("{}_60m.jp2".format(
                            selected_band)) >= 0:
                            study_bands[selected_band] = fich
                study_bands["B10"] = None
        for maska in [ "SNW", "CLD"]:
            
            for resolution in ["20", "60"]:
                files = glob.glob(os.path.join(granule_dir,
                                            "QI_DATA",
                                            "*%s*_%sm.jp2" 
                                            % (maska, resolution)))
                study_bands[maska+"_%s"%resolution] = files[0]
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
        
        hplot(boa_rho[~mask][::sub], toa_rho[~mask][::sub])

    def get_transform(self, the_date, band, mask="L2",
                      sub=10, nv=200, lw=2, odir='figures',
                      apply_model=False, plausible=True):

        # ensure odir exists
        if not os.path.exists(odir): os.makedirs(odir)

        fname = odir+'/'+'SEN2COR_%s_%s_%s_%s'%(self.site, self.tile, the_date.strftime("%Y-%m-%d_%H:%M"), band)

        toa_set = self.get_l1c_data(the_date)
        boa_set = self.get_l2_data(the_date)
        if toa_set is None or boa_set is None:
            print "No TILEs found for %s" % the_date
            return None
        g = gdal.Open(toa_set[TOA_list.index(band)])
        toa_rho = g.ReadAsArray()
        g = gdal.Open(boa_set[BOA_list.index(band)])
        boa_rho = g.ReadAsArray()
        if mask == "L2":
            print "Using L2A product mask"
            if band in ["B02", "B03", "B04", "B08"]:
                mask = self._get_mask(the_date, res="10")
            elif band in ["B05", "B06", "B07", "B11", "B12", "B8A"]:
                mask = self._get_mask(the_date, res="20")
            elif band in ["B01", "B09"]:
                mask = self._get_mask(the_date, res="60")
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
        
    def compare_boa_refl_MCD43(self, the_date, the_band):
        xstd = 30.
        ystd = 30.
        angle = 0.
        print "NOTE!! Gaussian not optimised for x- and y-shift!"
        psf = gaussian(xstd, ystd, angle, norm = True)
        mapping_arrays = Find_corresponding_pixels(ts.l2a_datasets[the_date][the_band+1])
        tile = mapping_arrays.keys()[0] # Only 1 tile...
        self.modis_files (tile, self.site)
        rho_boa_hires = gdal.Open(
            self.l2a_datasets[the_date][the_band]).ReadAsArray()/10000. # 
        rho_boa_s2 = PSF_convolve(rho_boa_hires, psf, mapping_arrays[tile][0], 
                                    mapping_arrays[tile][1])
        s2_mask = np.isfinite(rho_boa_s2)
        min_y, max_y = np.nonzero(s2_mask)[0].min(), \
            np.nonzero(s2_mask)[0].max()
        min_x, max_x = np.nonzero(s2_mask)[1].min(), \
            np.nonzero(s2_mask)[1].max()

        modis_prediction = self.predict_boa_refl_MCD43(the_date, the_band)
        return rho_boa_s2[min_y:max_y, min_x:max_x], \
            modis_prediction[min_y:max_y, min_x:max_x]
        


    def predict_boa_refl_MCD43(self, the_date, the_band):
        # find modis matching dataset
        print "FIX band_equivalences!!!!!!!!!!"
        band_equivalences = [ [[1, 1.]], [[2, 1.]], [[3, 1.]], 
                                [[4, 1.]], [[5, 1.]], [[6, 1.]], [[7, 1.]] ]
        sza, saa, vza, vaa = self.get_l1c_angles(the_date)
        raa = vaa - saa
        K = kernels.Kernels( vza, sza, raa,
            LiType='Sparse', doIntegrals=False, 
            normalise=1, RecipFlag=True, RossHS=False, MODISSPARSE=True,
            RossType='Thick' )
        kk = np.array([1., K.Ross[0], K.Li[0]])
        
        k = min(self.modis_filelist["MCD43A1"].keys(), 
                key=lambda x: abs(x - the_date))
        pred_boa_refl = np.zeros((2400, 2400))
        for band, w in band_equivalences[the_band]:
            
            g = gdal.Open('HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d' % 
                        (self.modis_filelist["MCD43A1"][k], band))
            kernel_weights = g.ReadAsArray()/10000.
            g = gdal.Open('HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Quality_Band%d' % 
                        (self.modis_filelist["MCD43A2"][k], band))
            qa = g.ReadAsArray()
            print "Only best quality MCD43 samples selected"
            kernel_weights = np.where(qa == 0, kernel_weights, np.nan)

            pred_boa_refl += w*np.sum(kernel_weights*kk[:, None, None], axis=0)
        return pred_boa_refl

    def modis_files (self, tile, site, folder="/storage/ucfajlg/S2_AC/MCD43/"):
        self.modis_filelist = {}
        for product in ["MCD43A1", "MCD43A2"]:
            self.modis_filelist[product] = {}
            files = glob.glob ( os.path.join( folder, site, 
                                             "%s*%s*hdf"%(product, tile)))
            
            for fich in files:
                date_str = os.path.basename(fich).split(".")[1]
                the_date = datetime.datetime.strptime(date_str, "A%Y%j")
                self.modis_filelist[product][the_date] = fich
            
if __name__ == "__main__":
    from spatial_mapping import *

    for (site,tile) in [ ["Ispra", "T32TMR"]]:
                        #["Pretoria_CSIR-DPSS", "35JPM"], 
                        #["Pretoria_CSIR-DPSS", "35JQM"]]:
    #for (site,tile) in [ 
                        #["Pretoria_CSIR-DPSS", "35JPM"], 
                        #["Pretoria_CSIR-DPSS", "35JQM"]]:

        ts = TeleSpazioComparison(site, tile)
        
        for ii, the_date in enumerate( ts.l2a_datasets.iterkeys()):        
            print the_date
            ts.compare_boa_refl_MCD43(the_date, 1)
            break
