import Isodata as iso
import Isodata2 as iso2
import numpy as np
import matplotlib.pyplot as plt

# Load Data Cube and converted data
print("Loading Data..")
cube = np.load("npIndian_pines.npy")
#data = ConvertDataCube(cube)
data = np.load("data.npy")

# Also play with data only containing spectral info
samps,features = data.shape
dataSpectral = np.ones((samps,features-2))
dataSpectral[:,:] = data[:,2:]

# Run isodata?
params = {"K": 15, "I" : 100, "P" : 2, "THETA_M" : 10, "THETA_S" : 0.1,"THETA_C" : 2, "THETA_O" : 0.01}
#gt = iso.isodata_classification(cube,parameters=params)
gt = iso2.isodata_classification(cube,parameters=params)

fig = plt.figure()
plt.imshow(gt)
plt.show()
