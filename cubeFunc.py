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

def RandIndex(classes,key): # Should be 1D arrays of equal length. The values in the array should be classes

	# https://en.wikipedia.org/wiki/Rand_index

	samples=len(classes)
	a=0
	b=0
	c=0
	d=0

	# Go through all pairs of data points
	for i in range(samples):
		for j in range(i,samples):

			if(classes[i] == classes[j]):
				if(key[i] == key[j] ):
					a += 1
				else:
					c += 1
			else:
				if(key[i] == key[j] ):
					d += 1
				else:
					b += 1

	return (a+b)/(a+b+c+d)



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