from unittest import TestCase
from work.utils import Distances

class TestUtilities(TestCase):
    def legacy_minkowski_distance(self, point1, point2):
        sum = 0
        for i in range(len(point1)):
            tmp = point1[i] - point2[i]
            tmp **=3
            sum += tmp

        return sum**(1/3)

    def AssertLists(self, left, right):
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertEqual(len(right), len(left))

        for i in range(len(left)):
            self.assertEqual(right[i], left[i])


class test_KNN_Tests(TestUtilities):
    def test_minkowski_distance_Test1(self):

        point1 = []
        point2 = []

        for p1, p2 in zip(point1, point2):
            self.assertEqual(self.legacy_minkowski_distance(p1,p2),
                             Distances.minkowski_distance(p1,p2))