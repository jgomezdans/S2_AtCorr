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
regex = r"B[0-9][0-9,A]?_ac"

BC_DIR = "/storage/ucfajlg/S2_AC/BC/ftp.brockmann-consult.de/data/May2017/products/"
MODIS_DIR = "/data/selene/ucfajlg/S2_AC/MCD43/"

TOA_list = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09",  "B11", "B12"]

BC_BOA_set = namedtuple("BC_BOA", 
                          "b1 b2 b3 b4 b5 b6 b7 b8 " + 
                          "b8a b11 b12 aot wvc o3 aero_type " + 
                          "status_10m")

BC_BOA_list = [ "B1", "B2", "B3", "B4", "B5", "B6", "B7",
                "B8", "B8A", "B11", "B12", "AOT", "WVC", "O3", "AERO_TYPE",
                "status_10m", ]

# B1, B8 and B10 are not in CNES dataset



class BrockmannComparison(TeleSpazioComparison):
    """A Class to do comparisons of the BC L2A product."""
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

        self.l2a_files = {}
        for cur, _dirs, files in os.walk(BC_DIR):
            if cur.find(self.tile) >= 0 and cur.endswith("-ac.data"):
                try:
                    key = datetime.datetime.strptime(os.path.basename(
                        cur).split("_")[2], "%Y%m%dT%H%M%S")
                except ValueError:
                    key = datetime.datetime.strptime(os.path.basename(
                        cur).split("_")[7], "V%Y%m%dT%H%M%S")
                key = key.replace(second=0)
                self.l2a_files[key] = os.path.join(cur)
                    
            
        
        for k in self.l2a_files.iterkeys():
            try:
                r = self.l1c_datasets[k]
                self.l2a_datasets[k] = self._bc_sorter( self.l2a_files[k])
            except KeyError:
                continue
    
    def _bc_sorter(self, path):
        
        granule = {}
        for cur, _dir, files in os.walk(path):
            for fich in files:
                if fich.endswith(".img"):
                    if re.search(regex, fich):
                        band_no = fich.split("_")[0] # (eg B02)
                        granule[band_no] = os.path.join(cur, fich)
                    elif fich.find("TC_Ozone") >= 0:
                        granule["O3"] = os.path.join(cur, fich)
                    elif fich.find("TC_WV") >= 0:
                        granule["WVC"] = os.path.join(cur, fich)
                    elif fich.find("AOD") >= 0:
                        granule["AOT"] = os.path.join(cur, fich)
                    elif fich.find("aerosol_type") >= 0:
                        granule["AERO_TYPE"] = os.path.join(cur, fich)
                    elif fich.find("status_10m") >= 0:
                        granule["status_10m"] = os.path.join(cur, fich)

        data = []
        for d in BC_BOA_list:
            if not os.path.exists(granule[d]):
                raise IOError
            data.append(granule[d])
        return BC_BOA_set(*data)
        
        
                
        

if __name__ == "__main__":

            
    #for (site,tile) in [["Pretoria", "35JPM"], ["Ispra", "T32TMR"], 
                        #["Pretoria", "35JQM"]]:
    for (site,tile) in [ ["Ispra", "T32TMR"]]:

        ts = BrockmannComparison(site, tile)
        
        for jj, the_date in enumerate(ts.l2a_datasets.iterkeys()):
            print the_date, ts.l1c_datasets[the_date]
            ts.compare_boa_refl_MCD43(the_date, 1)
            break
    
