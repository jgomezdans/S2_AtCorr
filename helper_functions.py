#!/usr/bin/env python
"""
Extract S2 L1, S2 L2, MODIS and Landsat data to do comparisons


"""
import os
import subprocess
import xml.etree.ElementTree as ET
import datetime


import numpy as np
import matplotlib.pyplot as plt
import gdal
import osr


def warp(args):
    # command construction with binary and options
    options = ['/opt/anaconda/bin/gdalwarp']
    options.extend(args)
    # call gdalwarp 
    subprocess.check_call(options)
    
    
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
    slave_ds = gdal.Open( layer % slave )
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
    master_geo = list(master_ds.GetGeoTransform())
    w = master_ds.RasterXSize
    h = master_ds.RasterYSize
    extent = (master_geo[0], 
              master_geo[3]+h*master_geo[-1],
              master_geo[0]+w*master_geo[1],
              master_geo[3])
    
    s = osr.SpatialReference(master_proj)
    epsg = s.GetAttrValue("AUTHORITY", 1)
    suffix = layer.split("_")[-1]
    output = slave.replace(".hdf", 
                           "_%s_cropped.vrt"%suffix)
    warp(['-overwrite', '-of', 'VRT', '-tap', '-tr', '480', '480',
          '-t_srs', "EPSG:%s"%epsg, '-te',
          '%f'%extent[0], '%f'%extent[1],'%f'%extent[2],
          '%f'%extent[3], layer%slave, output])
    return output

def load_emulator_training_set(
    emu_dir="/home/ucfajlg/Data/python/S2S3Synergy/optical_emulators"):
    import cPickle
    emulatorf = "isotropic_MSI_training_set_1100.npz"
    f = np.load(os.path.join(emu_dir, emulatorf))
    boa_refl = f['training_input'][:,-1]
    bands =[ "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12"]
    toa_refl = dict(zip(bands, f['training_output'].T))
    return boa_refl, toa_refl

def hplot(x,y,bar=True,log=True,image=0,new=True,thresh = 10,xlim=[0,1],ylim=[0,1],bins=[128,128]):
    if new:
      fig = plt.figure(figsize=(10,10))
      ax = plt.gca()
    else:
        ax = new
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
      im = ax.imshow(image,cmap='magma',extent=np.array(xyrange).flatten(),\
                   interpolation='nearest')
      ax.plot(xyrange[0],xyrange[1],'g--', lw=1.5)
      ##fig.colorbar(fraction=0.04)
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



def Find_corresponding_pixels(H_res_fname, destination_res=500):
    
    '''
    A function for the finding of corresponding pixels indexes 
    in H (sentinel 2) and L (MODIS) resolution image.
    
    args:
        H_res_fname -- the high resolution image filename, need to have geoinformation enbeded in the file
        destination_res -- for the calculation of pixel number in one MODIS tile
    return:
        index: a dictionary contain both the MODIS tile name and pixels indexes
    
    '''
    
    if destination_res % 250 == 0:
        pass
    else:
        print 'destination resolution can only be 250, 500 and 1000 !!!'
        raise IOError

    g = gdal.Open(H_res_fname)
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterXSize, g.RasterYSize

    wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    H_res_geo = osr.SpatialReference( )
    raster_wkt = g.GetProjection()
    H_res_geo.ImportFromWkt(raster_wkt)
    tx = osr.CoordinateTransformation(H_res_geo, wgs84)
    # so we need the four corners coordiates to check whether they are within the same modis tile
    (ul_lon, ul_lat, ulz ) = tx.TransformPoint( geo_t[0], geo_t[3])

    (lr_lon, lr_lat, lrz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
                                          geo_t[3] + geo_t[5]*y_size )

    (ll_lon, ll_lat, llz ) = tx.TransformPoint( geo_t[0] , \
                                          geo_t[3] + geo_t[5]*y_size )

    (ur_lon, ur_lat, urz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
                                          geo_t[3]  )

    #print (ul_lon, ul_lat), (lr_lon, lr_lat), (ll_lon, ll_lat), (ur_lon, ur_lat)
    # now its the s2 corners latitudes and longtitudes
    s_dic ={'UR_LAT': ur_lat,
           'UR_LON': ur_lon,
           'LR_LAT': lr_lat,
           'LR_LON': lr_lon,
           'UL_LAT': ul_lat,
           'UL_LON': ul_lon,
           'LL_LAT': ll_lat,
           'LL_LON': ll_lon}
    s_cors = ulx, uly, lrx, lry, llx, lly, urx, ury = 0,0,x_size, y_size, x_size,0, 0, y_size 
    s_corners = dict(zip(['ulx', 'uly', 'lrx', 'lry', 'llx', 'lly', 'urx', 'ury'], s_cors))
    
    a0, b0 = None, None
    corners = [(ul_lon, ul_lat), (lr_lon, lr_lat), (ll_lon, ll_lat), (ur_lon, ur_lat)]
    tiles = []
    for i,j  in enumerate(corners):
        h,v = mtile_cal(j[1], j[0])
        if (h==a0) &(v==b0):
            pass
        else:
            tiles.append([i,h,v]) # 0--ul;1--lr;2--ll;3--ur
            a0, b0 = h,v
    
    # The modis defults
    pix_num = 4800/(destination_res/250)
    
    inds = {}
    for i in tiles:
        latitudes, longtitudes = get_Mpix_wgs(h,v, pix_num).T[[1,0],]
        cors = bilineanr([latitudes, longtitudes], s_dic, s_corners)
        hinds =np.array([cors[0][(cors[0]>=0)&(cors[0]<x_size)&(cors[1]>=0)&(cors[1]<y_size)],
                     cors[1][(cors[0]>=0)&(cors[0]<x_size)&(cors[1]>=0)&(cors[1]<y_size)]]).astype(int)
        minds = np.where(((cors[0]>=0)&(cors[0]<x_size)&(cors[1]>=0)&(cors[1]<y_size)).reshape((pix_num,pix_num)))
        inds['h%02dv%02d'%(i[1],i[2])] = [hinds, np.array(minds)]

    return inds
