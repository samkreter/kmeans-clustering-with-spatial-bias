import numpy as np
import math

# This will convert a hyperspectral data cube into a 2D feature array.
# The row is the data point, the columns are the features.
# The first 2 features are the x and y position
# The rest of the features are the spectral components
def ConvertDataCube(cube):

	xsize,ysize,wavelengths = cube.shape
	numSamples = xsize * ysize
	numFeatures = wavelengths + 2

	data = np.ones((numSamples,numFeatures))

	for i in range(xsize):
		for j in range(ysize):

			data[i*ysize + j][0] = i
			data[i*ysize + j][1] = j

			for k in range(2,numFeatures):
				data[i*ysize + j][k] = cube[i][j][k-2]

	return data

# Converts the groundtruth (key) into 1D form compatible with the above features converted from the data cube
def ConvertGroundtruth(GT):
	xsize,ysize = GT.shape

	labels = np.ones(xsize*ysize)

	for i in range(xsize):
		for j in range(ysize):
			labels[i*ysize + j] = GT[i][j]

	return labels

# Computes the rand index of two sets of classes, also apparently found in scikit
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

# Computes the physical distance between the i'th and j'th data point
def SpatialDistanceEuclidean(data,i,j):
	return math.sqrt( (data[i][0]-data[j][0])^2 + (data[i][1]-data[j][1])^2 )

def SpatialDistanceL2(data,i,j):
	return (data[i][0]-data[j][0])^2 + (data[i][1]-data[j][1])^2 

# Area normalizes the spectra such that the spectra of each data points integrates to 1
def NormalizeSpectra(data):

	numSamples, numFeatures = data.shape

	for i in range(numSamples):
		area = 0
		for j in range(2,numFeatures):
			area += data[i][j]

		for j in range(2,numFeatures):
			data[i][j] /= area

	return data

# Uses the L2 norm to define a "distance" between two data points in terms of their spectral components
def SpectralDistanceL2(data,i,j):

	numSamples, numFeatures = data.shape

	L2 = 0

	for k in range(2,numFeatures):
		L2 += (data[i][k]-data[j][k])^2

	return L2






