import numpy as np

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
# length = 10
# classes = np.zeros(length)
# key = np.zeros(length)
# for i in range(length):
# 	classes[i] = 1
# 	key[i] = i >= 5
# print(RandIndex(classes,key))
# print(classes)
# print(key)


