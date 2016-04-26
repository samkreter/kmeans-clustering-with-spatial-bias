import scipy.io
import numpy as np

#convert main dataset
# mat = scipy.io.loadmat("Indian_pines.mat")
# npMat = np.array(mat['indian_pines'])

# np.save("npIndian_pines.npy",npMat)

#convert ground truth data set
mat = scipy.io.loadmat("Indian_pines_gt.mat")
npMat = np.array(mat['indian_pines_gt'])

np.save("npIndian_pines_gt.npy",npMat)