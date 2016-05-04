import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sklearn.metrics.pairwise as met
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import cluster

# This will convert a hyperspectral data cube into a 2D feature array.
# The row is the data point, the columns are the features.
# The first 2 features are the x and y position
# The rest of the features are the spectral components
def ConvertDataCube(cube):

	xsize,ysize,wavelengths = cube.shape
	numSamples = xsize * ysize
	numFeatures = wavelengths + 2

	data = np.ones((numSamples,numFeatures))

	print("Converting Data Cube...")
	for i in range(xsize):
		for j in range(ysize):

			data[i*ysize + j][0] = i
			data[i*ysize + j][1] = j

			for k in range(2,numFeatures):
				data[i*ysize + j][k] = cube[i][j][k-2]

	print("Done.")
	return data

# Converts the groundtruth (key) into 1D form compatible with the above features converted from the data cube
def ConvertGroundtruth(GT):
	xsize,ysize = GT.shape

	labels = np.ones(xsize*ysize)

	for i in range(xsize):
		for j in range(ysize):
			labels[i*ysize + j] = GT[i][j]

	return labels

# Inverse of the above operation. Assumes a square groundtruth
def ConvertLabels(labels):

	size = int(math.sqrt(len(labels)))

	gt = np.zeros((size,size))

	for i in range(size):
		for j in range(size):
			gt[i][j] = labels[i*size + j]

	return  gt

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
	return math.sqrt( (data[i][0]-data[j][0])**2 + (data[i][1]-data[j][1])**2 )

def SpatialDistanceL2(data,i,j):
	return (data[i][0]-data[j][0])**2 + (data[i][1]-data[j][1])**2 

# Area normalizes the spectra such that the spectra of each data points integrates to 1
def NormalizeSpectra(data):

	numSamples, numFeatures = data.shape

	for i in range(numSamples):
		area = sum(data[i,2:])
		data[i,:] /= area

	return data

# Uses the L2 norm to define a "distance" between two data points in terms of their spectral components
def SpectralDistanceL2(data,i,j):
	return sum((data[i,2:]-data[j,2:])**2)#/(numFeatures-2)**2

def WeightedDist_Arith(data,i,j,spatialFunc,spatialWeight,spectralFunc,spectralWeight):
	return ( spatialFunc(data,i,j)*spatialWeight + spectralFunc(data,i,j)*spectralWeight ) / (spatialWeight + spectralWeight)

def WeightedAffinity_Arith(data,i,j,sigma,spatialFunc,spatialWeight,spectralFunc,spectralWeight):
	return math.exp(-(( spatialFunc(data,i,j)*spatialWeight + spectralFunc(data,i,j)*spectralWeight ) / (spatialWeight + spectralWeight))**2 / (2*sigma**2))

def WeightedDist_Geo(data,i,j,spatialFunc,spatialWeight,spectralFunc,spectralWeight):
	return math.pow( (spatialFunc(data,i,j)**spatialWeight) *  (spectralFunc(data,i,j)**spectralWeight),1/(spatialWeight+spectralWeight))

# spatialWeight = 1
# spectralWeight = 1/1000
# sigma = 1
# def WeightedAffinity(vec1,vec2):
# 	return math.exp(-(( sum((vec1[:2]-vec2[:2])**2)*spatialWeight + sum((vec1[2:]-vec2[2:])**2)*spectralWeight ) / (spatialWeight+spectralWeight))**2 / (2*sigma))

def AdjMatrix(data,spatialFunc,spatialWeight,spectralFunc,spectralWeight):

	numSamples, numFeatures = data.shape

	mat = np.zeros((numSamples,numSamples))
	for i in range(numSamples):
		print(str(i+1)+" of "+str(numSamples))
		for j in range(i,numSamples):
			mat[i][j] = WeightedDist_Arith(data,i,j,spatialFunc,spatialWeight,spectralFunc,spectralWeight)
			mat[j][i] = mat[i][j]

	return mat

# Viewing Spectral Lines in the hyperspectral cube
def View(cube,groundtruth):

	xsize, ysize, wavelengths = cube.shape

	spectrum = np.zeros(wavelengths)
	old = np.zeros(wavelengths)

	x = 0
	y = 0

	for i in range(wavelengths):
		spectrum[i] = cube[x][y][i]

	im = plt.figure()
	plt.imshow(groundtruth)
	plt.scatter([x],[y])

	imSliceFig = plt.figure()

	fig = plt.figure()
	plt.ion()
	plt.plot(spectrum)
	plt.show()

	while(1):
		x,y = input('X Y:').split()
		layer = input('Layer:')

		x=int(x)
		y=int(y)
		layer=int(layer)

		if(x==-1 and y==-1):
			break

		x=max(0,min(xsize-1,x))
		y=max(0,min(ysize-1,y))

		for i in range(wavelengths):
			old[i] = spectrum[i]
			spectrum[i] = cube[x][y][i]

		imSlice = Slice(cube,layer)

		plt.figure(imSliceFig.number)
		imSliceFig.clf()
		plt.imshow(imSlice)

		plt.figure(fig.number)
		fig.clf()
		plt.plot(spectrum)
		plt.plot(old,'r')

		plt.figure(im.number)
		im.clf()
		plt.imshow(groundtruth)
		plt.scatter([x],[y])

def Slice(cube,layer):
	xsize,ysize,depth = cube.shape

	im = cube[:,:,layer]
	return im

def RandIndex(classes,key): # Should be 1D arrays of equal length. The values in the array should be classes

	# https://en.wikipedia.org/wiki/Rand_index

	if(len(classes) != len(key)):
		print( len(classes),len(key) )
		return -1

	samples=len(classes)
	a=0
	b=0
	c=0
	d=0

	# Go through all pairs of data points
	for i in range(samples):

		# Ignore Background in key
		if(key[i]==0):
			continue

		for j in range(i,samples):

			# Ignore Background in key
			if(key[j]==0):
				continue

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

def NeighborBias(sqrmap,maxClasses,radius):

	xsize,ysize = sqrmap.shape

	# Evolution of map
	newMap = np.zeros((xsize,ysize))

	neighborsClasses = np.zeros(maxClasses)
	neighborsHist = np.zeros(maxClasses)

	for i in range(xsize):
		for j in range(ysize):

			neighborsHist.fill(0)

			for ii in range(i-radius,i+radius+1):
				for jj in range(j-radius,j+radius+1):

					if(ii<0 or ii>=xsize or jj<0 or jj>=ysize):
						continue
					
					for k in range(maxClasses):
						if(neighborsClasses[k]==sqrmap[ii][jj]):
							neighborsHist[k]+=1
							break
						elif(k!=0 and neighborsClasses[k]==0):
							neighborsClasses[k] = sqrmap[ii][jj]
							neighborsHist[k]+=1
							break

			newMap[i][j] = neighborsClasses[neighborsHist.argmax()]
			# print(neighborsClasses)
			# print(neighborsHist)
			# print(newMap[i][j])
			# print()

	return newMap

#####################################################
		# DO IT
#####################################################

# PARAMETERS
spectralScale = 1/1000
PCAcomps = 10
numClasses = 12
DB_eps = 0.7
whitening = True

# Load Data Cube and converted data
print("Loading Data..")
cube = np.load("npIndian_pines.npy")
#data = ConvertDataCube(cube)
data = np.load("data.npy")

# Also play with data only containing spectral info
samps,features = data.shape
dataSpectral = np.ones((samps,features-2))
dataSpectral[:,:] = data[:,2:]

# Data Scaling
data[2:] *= spectralScale

# Load Ground Truth
gt = np.load("npIndian_pines_gt.npy")
key = ConvertGroundtruth(gt)

# PCA Dimensionality Reduction
print("Doing PCA..")
pca = PCA(n_components=PCAcomps,whiten=whitening)
#dataPCA = pca.fit_transform(data)
dataPCA = pca.fit_transform(dataSpectral)

# PCA Stats
numSamps,numFeatures = dataPCA.shape
print(pca.explained_variance_ratio_) 
print(pca.components_)

# Look at PCA Components
# comp = pca.components_[0,:]
# plt.figure()
# plt.ion()
# plt.plot(comp)
# plt.show()
# input()

# K Means
# print("K-Meansing..")
# k_means = KMeans(n_clusters=17)
# k_means.fit(dataPCA)
# labels = k_means.labels_

# Spectral Clustering (rbf) DONT DO IT TAKES TOO LONG AND CRASHES
# print("Doing Spectral Clustering..")
# spectral = cluster.SpectralClustering(n_clusters=17)
# spectral.fit(dataPCA)
# labels = spectral.labels_

# DBSCAN
print("Doing DBSCAN..")
DB = cluster.DBSCAN(eps=DB_eps,min_samples=numClasses)#,metric=WeightedAffinity)
DB.fit(dataPCA)
labels = DB.labels_ 

# Neighborhood Biasing
evo0 = ConvertLabels(labels)
evo1 = NeighborBias(evo0,numClasses,1)
labels = ConvertGroundtruth(evo1)

# Rand Index
print("Calculating Rand Index..")
#print(RandIndex(labels,key))
print(adjusted_rand_score(labels,key))

# Visualize Ground Truth Prediction
plt.ion()
fig = plt.figure()
plt.imshow(gt)
plt.show()

fig2 = plt.figure()
plt.imshow(evo0)
plt.show()

fig3 = plt.figure()
plt.imshow(evo1)
plt.show()


#Testing stuff
# a = np.array([1,2,3,4])
# b = np.array([5,4,3,2])
# print(sum((-b[:2]+a[:2])**2))



input()


# start = time.time()

#a = np.random.rand(1000,256)
#print( WeightedDist_Arith(data,12,13,SpatialDistanceEuclidean,0,SpectralDistanceL2,1) )
#dists = met.pairwise_distances(a,n_jobs=-1)

# end = time.time()
#print("Time:"+str(end-start))

# xsize,ysize,depth = cube.shape
# refl = np.zeros((xsize,ysize))
# for i in range(xsize):
# 	for j in range(ysize):
# 		for k in range(depth):
# 			refl[i][j]+=cube[i][j][k]
# reflim = plt.figure()
# plt.imshow(refl)

#View(cube,gt)











