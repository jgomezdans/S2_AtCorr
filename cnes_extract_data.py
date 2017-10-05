#!/usr/bin/env python
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

from extract_data import TeleSpazioComparison

parent_folder = "/data/selene/ucfajlg/S2_AC/TeleSpazio/" + \
                "ftp.telespazio.fr/outgoing/L2A_Products/"

CNES_DIR = "/data/selene/ucfajlg/S2_AC/CNES/V1_1/"
MODIS_DIR = "/data/selene/ucfajlg/S2_AC/MCD43/"


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
                                "%Y%m%d-%H%M%S-%f").replace(microsecond=0)
                print "Dealing with iamge acquired on %s" % key
                
                x = self._unpack_data(granule)
                self.l2a_files[key] = x
       
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
        
if __name__ == "__main__":
    from spatial_mapping import *

    for (site,tile) in [ ["Ispra", "T32TMR"]]:
                        #["Pretoria_CSIR-DPSS", "35JPM"], 
                        #["Pretoria_CSIR-DPSS", "35JQM"]]:

        ts = CNESComparison(site, tile)
        
        for ii, the_date in enumerate( ts.l2a_datasets.iterkeys()):        
            print the_date
            ts.compare_boa_refl_MCD43(the_date, 1)
            break
