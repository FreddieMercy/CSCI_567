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
    def test_read_column_Test1(self):
        alpha = np.array([
            np.array([63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0, 0]),
            np.array([67.0, 1.0, 4.0, 160.0, 286.0, 0.0, 2.0, 108.0, 1.0, 1.5, 2.0, 3.0, 3.0, 1]),
            np.array([37.0, 1.0, 3.0, 130.0, 250.0, 0.0, 0.0, 187.0, 0.0, 3.5, 3.0, 0.0, 3.0, 0]),
            np.array([67.0, 1.0, 4.0, 120.0, 229.0, 0.0, 2.0, 129.0, 1.0, 2.6, 2.0, 2.0, 7.0, 1]),
            np.array([41.0, 0.0, 2.0, 130.0, 204.0, 0.0, 2.0, 172.0, 0.0, 1.4, 1.0, 0.0, 3.0, 0]),
            np.array([56.0, 1.0, 2.0, 120.0, 236.0, 0.0, 0.0, 178.0, 0.0, 0.8, 1.0, 0.0, 3.0, 0]),
            np.array([62.0, 0.0, 4.0, 140.0, 268.0, 0.0, 2.0, 160.0, 0.0, 3.6, 3.0, 2.0, 3.0, 1]),
            np.array([57.0, 0.0, 4.0, 120.0, 354.0, 0.0, 0.0, 163.0, 1.0, 0.6, 1.0, 0.0, 3.0, 0]),
            np.array([63.0, 1.0, 4.0, 130.0, 254.0, 0.0, 2.0, 147.0, 0.0, 1.4, 2.0, 1.0, 7.0, 1]),
            np.array([53.0, 1.0, 4.0, 140.0, 203.0, 1.0, 2.0, 155.0, 1.0, 3.1, 3.0, 0.0, 7.0, 1]),
            np.array([57.0, 1.0, 4.0, 140.0, 192.0, 0.0, 0.0, 148.0, 0.0, 0.4, 2.0, 0.0, 6.0, 0])])

        print(alpha[:, 0])
