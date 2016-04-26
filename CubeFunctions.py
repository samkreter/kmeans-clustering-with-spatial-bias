import numpy as np
import math

def EuclidDistance(features,i,j):
	return math.sqrt( (features[i][0]-features[j][0])^2 + (features[i][1]-features[j][1])^2 )