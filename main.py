from clusteringUtils import *

if __name__ == '__main__':

    #load the data to use
    cube = np.load("npIndian_pines.npy")
    #get the unrolled data for the kmeans
    data = np.load("data.npy")

    #get the ground truth
    gt = np.load("npIndian_pines_gt.npy")
    #convert
    key = ConvertGroundtruth(gt)

    labels = kMeansMaxSpectralWeight(cube,data,gt,key)

    print("Doing Neighborhood Biasing..")
    evo0 = ConvertLabels(labels)
    evo1 = NeighborBias(evo0,numClasses,1)

    i=0
    for i in range(10):
        evo2 = NeighborBias(evo1,numClasses,1)
        if np.array_equal(evo1,evo2):
            print("Converged!")
        evo1 = evo2

    labels = ConvertGroundtruth(evo2)

    # Rand Index
    print("Calculating Rand Index..")
    print(RandIndex(labels,key))

    # Adjusted rand index
    print("Calculating Adjusted Rand Index..")
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

    #Hold so you can see the graphs at the end
    input()
