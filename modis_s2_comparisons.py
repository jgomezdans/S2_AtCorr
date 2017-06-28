#!/usr/bin/env python
"""

"""
from extract_data import *
from helper_functions import *
import pickle
from kernels import Kernels

try:
    assert ts
except:
    ts = TeleSpazioComparison("Ispra", "T32TMR")
    for ii, the_date in enumerate( ts.l1c_files.iterkeys()):
        print ts.get_l1c_data(the_date)
        if ii == 5:
            break
    # modis mcd43 datasets
    modis_times = ts.get_modis_files("Ispra")

MCD43A2 =  modis_times['MCD43A2'][the_date]
MCD43A1 =  modis_times['MCD43A1'][the_date]

# need to know which bands for the mapping
mapping = pickle.load(open('transfer_functions/mappings.dump','r'))

s2map = mapping['sentinel2']

# read s2 data
sza,saa,vza,vaa = ts.get_l1c_angles(the_date)
print sza,saa,vza,vaa
# get kernels
K = Kernels([vza], [sza], [saa-vaa], LiType="Sparse", doIntegrals=False,\
                        normalise=1, RecipFlag=True,\
                        RossHS=False, MODISSPARSE=True, RossType="Thick",nbar=0.)
k0 = 1.0
k1 = K.Ross[0]
k2 = K.Li[0]
kk = np.array([k0,k1,k2])[:,np.newaxis, np.newaxis]

#j is the S2 band
s2sim = []

ns2 = len(s2map)
j = 0
modis_bands = s2map[j]['terra']['bandsB']
nb = modis_bands.shape[0]
# read MODIS data
mcd43 = []
mask = 0
for i in xrange(nb):
    data = gdal.Open(MCD43A1[i]).ReadAsArray()
    mask = mask + (data[0] == 32767)
    if len(mcd43) == 0:
        # intercept
        mcd43.append(np.ones_like(data[0]).astype(float))
    
    data = (kk * data/1000.).sum(axis=0)
    mcd43.append(data)
mcd43 = np.array(mcd43)
mask = mask>=1
mcd43[:,mask] = np.nan
M = s2map[j]['terra']['beta_hat']

s2sim.append(np.einsum('bi,bxy->xy',M,mcd43))

s2data = ts.get_l2_data(the_date)

