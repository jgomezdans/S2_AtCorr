#!/usr/bin/env python
"""
Extract S2 L1, S2 L2, MODIS and Landsat data to do comparisons


"""
import xml.etree.ElementTree as ET
import datetime
import glob
import os
import sys
from collections import namedtuple


import numpy as np
import matplotlib.pyplot as plt
import gdal

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


def reproject_image_to_master ( master, slave, layer, 
                               res=None ):
    """This function reprojects an image (``slave``) to
    match the extent, resolution and projection of another
    (``master``) using GDAL. The newly reprojected image
    is a GDAL VRT file for efficiency. A different spatial
    resolution can be chosen by specifyign the optional
    ``res`` parameter. The function returns the new file's
    name.
    Parameters
    -------------
    master: str 
        A filename (with full path if required) with the 
        master image (that that will be taken as a reference)
    slave: str 
        A filename (with path if needed) with the image
        that will be reprojected
    res: float, optional
        The desired output spatial resolution, if different 
        to the one in ``master``.
    Returns
    ----------
    The reprojected filename

    """
    slave_ds = gdal.Open( slave )
    if slave_ds is None:
        raise IOError, "GDAL could not open slave file %s " \
            % slave
    slave_proj = slave_ds.GetProjection()
    slave_geotrans = slave_ds.GetGeoTransform()
    data_type = slave_ds.GetRasterBand(1).DataType
    n_bands = slave_ds.RasterCount

    master_ds = gdal.Open( master )
    if master_ds is None:
        raise IOError, "GDAL could not open master file %s " \
            % master
    master_proj = master_ds.GetProjection()
    master_geotrans = master_ds.GetGeoTransform()
    w = master_ds.RasterXSize
    h = master_ds.RasterYSize
    if res is not None:
        master_geotrans[1] = float( res )
        master_geotrans[-1] = - float ( res )

    dst_filename = slave.replace( ".tif", "_crop.vrt" )
    dst_ds = gdal.GetDriverByName('VRT').Create(dst_filename,
                                                w, h, n_bands, data_type)
    dst_ds.SetGeoTransform( master_geotrans )
    dst_ds.SetProjection( master_proj)

    gdal.ReprojectImage( slave_ds, dst_ds, slave_proj,
                         master_proj, gdal.GRA_NearestNeighbour)
    dst_ds = None  # Flush to disk
    return dst_filename


def hplot(x,y,bar=True,log=True,image=0,new=True,thresh = 10,xlim=[0,1],ylim=[0,1],bins=[128,128]):
    if new:
      plt.figure(figsize=(10,10))
    xyrange = [x.min(),x.max()],[y.min(),y.max()]
    xyrange = xlim,ylim
    min,max = np.min([x.min(),y.min()]),np.max([x.max(),y.max()])

    #xyrange = [0,max],[0,max]
    hh, locx, locy = np.histogram2d( x, y, range=xyrange,
                                    bins=bins)

    hh[hh<thresh] = np.nan
    if log:
      hh = np.log(hh)
    image += np.flipud(hh.T)

    if bar:
      im = plt.imshow(image,cmap='magma',extent=np.array(xyrange).flatten(),\
                   interpolation='nearest')
      plt.plot(xyrange[0],xyrange[1],'w--')
      plt.colorbar(fraction=0.04)
    return image

def parse_xml(filename):
    """Parses the XML metadata file to extract view/incidence 
    angles. The file has grids and all sorts of stuff, but
    here we just average everything, and you get 
    1. SZA
    2. SAA 
    3. VZA
    4. VAA.
    """
    with open(filename, 'r') as f:
        tree = ET.parse(filename)
        root = tree.getroot()

        vza = []
        vaa = []
        for child in root:
            for x in child.findall("Tile_Angles"):
                for y in x.find("Mean_Sun_Angle"):
                    if y.tag == "ZENITH_ANGLE":
                        sza = float(y.text)
                    elif y.tag == "AZIMUTH_ANGLE":
                        saa = float(y.text)
                for s in x.find("Mean_Viewing_Incidence_Angle_List"):
                    for r in s:
                        if r.tag == "ZENITH_ANGLE":
                            vza.append(float(r.text))
                            
                        elif r.tag == "AZIMUTH_ANGLE":
                            vaa.append(float(r.text))
                            
    return sza, saa, np.mean(vza), np.mean(vaa)

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
        angles = parse(os.path.join(granule_dir,xml_file))
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
        
    def do_scatter(self, the_date, band):
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
        hplot(boa_rho[~mask][::10], toa_rho[~mask][::10])
        
    def get_modis_files(self, site):
        """Gets the MODIS files. You get in return a dictionary
        with the same keys as the L1 and L2 datasets, with the 
        filenames for the MCD43A1 and MCD43A2 products.
        """
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
            
            for s2_date in self.l2a_files.iterkeys():
                for i, modis_date in enumerate(modis_dates):
                    if modis_date.date() == s2_date.date():
                        modis_mapper[product][s2_date] = \
                            files[i]
        return modis_mapper

if __name__ == "__main__":
    ts = TeleSpazioComparison("Ispra", "T32TMR")
    for ii, the_date in enumerate( ts.l1c_files.iterkeys()):
        print ts.get_l1c_data(the_date)
        if ii == 5:
            break
    ts.do_scatter(the_date, "B02")
    #modis_times = ts.get_modis_files("Ispra")
        
