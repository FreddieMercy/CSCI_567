import numpy as np


#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def dot_product_square_of_self(self):
    return np.dot(self, self.T)


def square_element_wise(matrix):
    return [dot_product_square_of_self(i) for i in matrix]


def get_k_means_plus_plus_center_indices(N, n_cluster, X, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, N)  # this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
    # according to some distribution, first generate a random number between 0 and
    # 1 using generator.rand(), then find the the smallest index n so that the
    # cumulative probability from example 1 to example n is larger than r.
    #############################################################################

    centers = [p]

    for i in range(n_cluster - 1):
        mu = []
        for n in range(N):
            if not n in centers:
                ds = []
                for point in centers:
                    ds.append(dot_product_square_of_self(X[n] - X[point]))
                mu.append(min(ds))
            else:
                mu.append(-1)
        centers.append(np.argmax(np.array(mu)))
    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################

        def calc_distortion_objective(X, centers):
            return np.mean([np.min(square_element_wise(centers - X[n])) for n in range(N)])

        distortion = 0
        centroids = np.array([x[n] for n in self.centers])
        Y = np.array([np.argmin(square_element_wise(centroids - x[n])) for n in range(N)])

        for i in range(self.max_iter):
            local_distortion = calc_distortion_objective(x, centroids)
            if abs(distortion - local_distortion) <= self.e:
                return centroids, Y, i

            group_by_k = {}
            for n in range(N):
                group_by_k.get(Y[n], []).append(x[n])

            for k, items in group_by_k.items():
                centroids[k] = np.mean(np.array(items), axis=1)

            Y = np.array([np.argmin(square_element_wise(centroids - x[n])) for n in range(N)])

        return centroids, Y, self.max_iter


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################

        K_means = KMeans(len(x), self.n_cluster, x, self.generator)
        centroids, membership, _ = K_means.fit(x, centroid_func)
        votes = np.array([np.zeros(self.n_cluster) for i in range(self.n_cluster)])

        for predict, actual in zip(membership, y):
            votes[predict][actual] += 1

        centroid_labels = np.array([0 for i in range(self.n_cluster)])

        for i in range(self.n_cluster):
            centroid_labels[i] = np.argmax(votes[i])

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################

        return np.array([self.centroid_labels[np.argmin(square_element_wise(self.centroids - x[n]))] for n in range(N)])


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            image[row][col] = code_vectors[np.argmin(square_element_wise(code_vectors - image[row][col]))]

    return image
