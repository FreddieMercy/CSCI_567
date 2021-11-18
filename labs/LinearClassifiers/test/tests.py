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
        X = [

            [67.0, 1.0, 1],
            [69.0, 1.0, 0],
            [45.0, 0.0, 0],
            [50.0, 0.0, 0],
            [59.0, 1.0, 1],
            [50.0, 0.0, 0],
            [64.0, 0.0, 0],
            [57.0, 1.0, 0],
            [64.0, 0.0, 0],
            [43.0, 1.0, 0],
            [45.0, 1.0, 1],
            [58.0, 1.0, 1],
            [50.0, 1.0, 1],
            [55.0, 1.0, 0],
            [62.0, 0.0, 1],
            [37.0, 0.0, 0],
            [38.0, 1.0, 1],
            [41.0, 1.0, 0],
            [66.0, 0.0, 1],
            [52.0, 1.0, 1]
        ]

        y = [

            [1],
            [0],
            [0],
            [0],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [1]


        ]

