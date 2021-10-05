import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """

    tp = 0
    falses = 0
    for r, p in zip(real_labels, predicted_labels):
        if r == p and r == 1:
            tp+=1
        if r != p:
            falses+=1

    return tp / (tp + 0.5 * falses)


class Distances:
    @staticmethod
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        return np.cbrt(sum([abs(p1 - p2) ** 3 for p1, p2 in zip(point1, point2)]))

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.sqrt(sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]))

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        x1 = np.sqrt(np.dot(point1, point1))
        x2 = np.sqrt(np.dot(point2, point2))

        if x1 == 0 or x2 == 0:
            return 1
        return 1 - (np.dot(point1, point2) / (x1 * x2))


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scalar = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """

        f_score = None

        for k in range(1, len(x_train)):
            for key in distance_funcs:
                knn = KNN(k, distance_funcs[key])

                knn.train(x_train, y_train)

                new_f_score = f1_score(y_val, knn.predict(x_val))

                if f_score is None or new_f_score > f_score:
                    self.best_k = k
                    self.best_distance_function = distance_funcs[key]
                    self.best_model = knn
                    f_score = new_f_score

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """

        f_score = None
        
        for k in range(1, len(x_train)):
            for key in distance_funcs:
                for scalarName in scaling_classes:
                    knn = KNN(k, distance_funcs[key])

                    scalar = scaling_classes[scalarName]()

                    knn.train(scalar(x_train), y_train)

                    new_f_score = f1_score(y_val, knn.predict(scalar(x_val)))

                    if f_score is None or new_f_score > f_score:
                        self.best_k = k
                        self.best_distance_function = distance_funcs[key]
                        self.best_scalar = scalar
                        self.best_model = knn
                        f_score = new_f_score

class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, feature):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        features = [list(f) for f in feature]

        for i in range(len(features)):

            deno = np.sqrt(np.dot(features[i], features[i]))

            if deno == 0:
                continue

            features[i] = [p / deno for p in features[i]]

        return features


class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, feature):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        features = [list(f) for f in feature]

        globalMini = [min(features[:][i]) for i in range(len(features[0]))]
        globalMaxi = [max(features[:][i]) for i in range(len(features[0]))]

        for i in range(len(features)):
            for j in range(len(features[0])):

                if globalMaxi[j] == globalMini[j]:
                    features[i][j] = 0
                    continue

                features[i][j] = (features[i][j] - globalMini[j]) / (globalMaxi[j] - globalMini[j])

        return features
