import scipy.io
import numpy as np


mat = scipy.io.loadmat("Indian_pines.mat")
npMat = np.array(mat['indian_pines'])

np.save("npIndian_pines.npy",npMat)
