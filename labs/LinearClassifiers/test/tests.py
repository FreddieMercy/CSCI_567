from unittest import TestCase

import numpy as np


class TestUtilities(TestCase):
    def legacy_minkowski_distance(self, point1, point2):
        sum = 0
        for i in range(len(point1)):
            tmp = abs(point1[i] - point2[i])
            tmp **= 3
            sum += tmp
        return np.cbrt(sum)

    def legacy_euclidean_distance(self, point1, point2):
        sum = 0
        for i in range(len(point1)):
            tmp = point1[i] - point2[i]
            tmp **= 2
            sum += tmp
        return np.sqrt(sum)

    def AssertLists(self, left, right):
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertEqual(len(right), len(left))

        for i in range(len(left)):
            self.assertEqual(right[i], left[i])


class test_KNN_Tests(TestUtilities):

    def test_perceptron_Test1(self):
        X = np.array([

            np.array([67.0, 1.0, 1]),
            np.array([69.0, 1.0, 0]),
            np.array([45.0, 0.0, 0]),
            np.array([50.0, 0.0, 0]),
            np.array([59.0, 1.0, 1]),
            np.array([50.0, 0.0, 0]),
            np.array([64.0, 0.0, 0]),
            np.array([57.0, 1.0, 0]),
            np.array([64.0, 0.0, 0]),
            np.array([43.0, 1.0, 0]),
            np.array([45.0, 1.0, 1]),
            np.array([58.0, 1.0, 1]),
            np.array([50.0, 1.0, 1]),
            np.array([55.0, 1.0, 0]),
            np.array([62.0, 0.0, 1]),
            np.array([37.0, 0.0, 0]),
            np.array([38.0, 1.0, 1]),
            np.array([41.0, 1.0, 0]),
            np.array([66.0, 0.0, 1]),
            np.array([52.0, 1.0, 1])
        ])

        w = np.array([1, -1, 0])
        b = 100

        Y = np.dot(X, w) + b

        y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1])

        y = np.array([-1 if i == 0 else i for i in y])

        tmp = max(0, y*Y)

        print()
