import numpy as np

# This will convert a hyperspectral data cube into a 2D feature array. 
# The row is the data point, the columns are the features.
# The first 2 features are the x and y position
# The rest of the features are the spectral components
def ConvertDataCube(cube):

	xsize,ysize,wavelengths = cube.shape
	numSamples = xsize * ysize
	numFeatures = wavelengths + 2

	features = np.ones((numSamples,numFeatures))

	for i in range(xsize):
		for j in range(ysize):

			features[i*ysize + j][0] = i
			features[i*ysize + j][1] = j

			for k in range(2,numFeatures):
				features[i*ysize + j][k] = cube[i][j][k-2]

	return features

# Testing
# xsize = 10
# ysize = 3
# depth = 4

# cube = np.zeros((xsize,ysize,depth))

# for i in range(xsize):
# 	for j in range(ysize):
# 		for k in range(depth):
# 			cube[i][j][k] = k

# print(ConvertDataCube(cube))