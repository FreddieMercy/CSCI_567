from unittest import TestCase

from homework3 import BFS, Utilities, UCS


class TestUtilities(TestCase):
    def ParseAndSearch(self, data, search):
        dimensions = Utilities.ConvertToInt(data[1])
        start = Utilities.ConvertToInt(data[2])
        end = Utilities.ConvertToInt(data[3])
        data = [Utilities.ConvertToInt(line) for line in data[5:]]

        return search.Evaluate(start, end, Utilities.ParseActions(data))

    def AssertLists(self, left, right):
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertEqual(len(right), len(left))

        for i in range(len(left)):
            self.assertEqual(right[i], left[i])


class test_BFS_Tests(TestUtilities):
    def test_BFS_Test1(self):
        data = [
            "BFS",
            "10 10 10",
            "1 3 1",
            "5 3 4",
            "13",
            "1 3 1 5",
            "1 3 2 1 6 11",
            "2 3 2 2 5 11",
            "2 3 3 6 14 16",
            "2 4 2 11 17",
            "3 2 2 3 15",
            "3 2 4 7 16",
            "3 3 2 4 5",
            "3 3 3 6 14 17 18",
            "3 4 3 14",
            "4 3 4 1 10 11",
            "5 3 4 2 5",
            "5 3 5 6 14"
        ]

        expect = [
            "6",
            "7",
            "1 3 1 0",
            "1 3 2 1",
            "2 3 2 1",
            "3 3 3 1",
            "3 2 4 1",
            "4 3 4 1",
            "5 3 4 1"
        ]

        output = self.ParseAndSearch(data, BFS())
        self.AssertLists(output, expect)

    def test_BFS_Test2(self):
        data = [
            "BFS",
            "4 4 4",
            "0 0 0",
            "2 3 1",
            "8",
            "0 0 0 7 11",
            "1 0 1 7 14",
            "1 1 0 7 10",
            "2 1 1 10 15",
            "2 2 0 5 10",
            "2 2 16",
            "2 2 2 3 18",
            "2 3 2 4"
        ]

        expect = [
            "FAIL"
        ]

        output = self.ParseAndSearch(data, BFS())
        self.AssertLists(output, expect)

    def test_BFS_Test3(self):
        data = [
            "BFS",
            "10 10 10",
            "7 0 1",
            "5 2 3",
            "11",
            "5 1 3 3 16",
            "5 2 2 5 8 17",
            "5 2 3 1 4 6",
            "6 1 1 3 5 8 15",
            "6 1 2 3 6 9",
            "6 2 1 4 8",
            "6 2 2 4 18",
            "6 2 3 2",
            "7 0 1 1 9",
            "7 1 1 8 9",
            "8 0 1 2 9"
        ]

        expect = [
            "4",
            "5",
            "7 0 1 0",
            "6 1 1 1",
            "6 1 2 1",
            "5 2 2 1",
            "5 2 3 1"
        ]

        output = self.ParseAndSearch(data, BFS())
        self.AssertLists(output, expect)

    def test_BFS_Test4(self):
        data = [
            "BFS",
            "5 5 5",
            "1 0 4",
            "3 1 2",
            "12",
            "0 0 1 1",
            "0 3 3 8",
            "1 0 1 2",
            "1 0 4 3 16",
            "1 1 3 1 3 5 12 16 17",
            "1 1 4 4 6 12",
            "1 2 2 17",
            "1 2 3 4 9",
            "2 1 2 1 12 13",
            "2 1 3 2 13",
            "3 1 1 13 5",
            "3 1 2 2 6"
        ]

        expect = [
            "3",
            "4",
            "1 0 4 0",
            "1 1 3 1",
            "2 1 2 1",
            "3 1 2 1"
        ]

        output = self.ParseAndSearch(data, BFS())
        self.AssertLists(output, expect)


class test_UCS_Tests(TestUtilities):
    def test_UCS_Test1(self):
        data = [
            "UCS",
            "10 10 10",
            "7 0 1",
            "5 2 3",
            "11",
            "5 1 3 3 16",
            "5 2 2 5 8 17",
            "5 2 3 1 4 6",
            "6 1 1 3 5 8 15",
            "6 1 2 3 6 9",
            "6 2 1 4 8",
            "6 2 2 4 18",
            "6 2 3 2",
            "7 0 1 1 9",
            "7 1 1 8 9",
            "8 0 1 2 9"
        ]

        expect = [
            "48",
            "5",
            "7 0 1 0",
            "6 1 1 14",
            "6 1 2 10",
            "5 2 2 14",
            "5 2 3 10"
        ]

        output = self.ParseAndSearch(data, UCS())
        self.AssertLists(output, expect)

    def test_UCS_Test2(self):
        data = [
            "UCS",
            "5 5 5",
            "1 0 4",
            "3 1 2",
            "12",
            "0 0 1 1",
            "0 3 3 8",
            "1 0 1 2",
            "1 0 4 3 16",
            "1 1 3 1 3 5 12 16 17",
            "1 1 4 4 6 12",
            "1 2 2 17",
            "1 2 3 4 9",
            "2 1 2 1 12 13",
            "2 1 3 2 13",
            "3 1 1 13 5",
            "3 1 2 2 6"
        ]

        expect = [
            "38",
            "4",
            "1 0 4 0",
            "1 1 3 14",
            "2 1 2 14",
            "3 1 2 10"
        ]

        output = self.ParseAndSearch(data, UCS())
        self.AssertLists(output, expect)

    def test_UCS_Test3(self):
        data = [
            "UCS",
            "4 4 4",
            "0 0 0",
            "2 3 1",
            "8",
            "0 0 0 7 11",
            "1 0 1 7 14",
            "1 1 0 7 10",
            "2 1 1 10 15",
            "2 2 0 5 10",
            "2 2 16",
            "2 2 2 3 18",
            "2 3 2 4"
        ]

        expect = [
            "FAIL"
        ]

        output = self.ParseAndSearch(data, UCS())
        self.AssertLists(output, expect)

    def test_UCS_Test4(self):
        data = [
            "UCS",
            "10 10 10",
            "1 3 1",
            "5 3 4",
            "13",
            "1 3 1 5",
            "1 3 2 1 6 11",
            "2 3 2 2 5 11",
            "2 3 3 6 14 16",
            "2 4 2 11 17",
            "3 2 2 3 15",
            "3 2 4 7 16",
            "3 3 2 4 5",
            "3 3 3 6 14 17 18",
            "    3 4 3 14",
            "   4 3 4 1 10 11",
            "    5 3 4 2 5",
            "    5 3 5 6 14"
        ]

        expect = [
            "72",
            "7",
            "1 3 1 0",
            "1 3 2 10",
            "2 3 2 10",
            "3 3 3 14",
            "3 2 4 14",
            "4 3 4 14",
            "5 3 4 10"
        ]

        output = self.ParseAndSearch(data, UCS())
        self.AssertLists(output, expect)


class test_Astar_Tests(TestUtilities):
    def test_Astar_Test1(self):
        data = [
            "A*",
            "10 10 10",
            "7 0 1",
            "5 2 3",
            "11",
            "5 1 3 3 16",
            "5 2 2 5 8 17",
            "5 2 3 1 4 6",
            "6 1 1 3 5 8 15",
            "6 1 2 3 6 9",
            "6 2 1 4 8",
            "6 2 2 4 18",
            "6 2 3 2",
            "7 0 1 1 9",
            "7 1 1 8 9",
            "8 0 1 2 9"
        ]

        expect = [
            "48",
            "5",
            "7 0 1 0",
            "6 1 1 14",
            "6 1 2 10",
            "5 2 2 14",
            "5 2 3 10"
        ]

        output = self.ParseAndSearch(data, UCS())
        self.AssertLists(output, expect)

    def test_Astar_Test2(self):
        data = [
            "A*",
            "4 4 4",
            "0 0 0",
            "2 3 1",
            "8",
            "0 0 0 7 11",
            "1 0 1 7 14",
            "1 1 0 7 10",
            "2 1 1 10 15",
            "2 2 0 5 10",
            "2 2 16",
            "2 2 2 3 18",
            "2 3 2 4"
        ]

        expect = [
            "FAIL"
        ]

        output = self.ParseAndSearch(data, UCS())
        self.AssertLists(output, expect)

    def test_Astar_Test3(self):
        data = [
            "A*",
            "10 10 10",
            "1 3 1",
            "5 3 4",
            "13",
            "1 3 1 5",
            "1 3 2 1 6 11",
            "2 3 2 2 5 11",
            "2 3 3 6 14 16",
            "2 4 2 11 17",
            "3 2 2 3 15",
            "3 2 4 7 16",
            "3 3 2 4 5",
            "3 3 3 6 14 17 18",
            "    3 4 3 14",
            "   4 3 4 1 10 11",
            "    5 3 4 2 5",
            "    5 3 5 6 14"
        ]

        expect = [
            "72",
            "7",
            "1 3 1 0",
            "1 3 2 10",
            "2 3 2 10",
            "3 3 3 14",
            "3 2 4 14",
            "4 3 4 14",
            "5 3 4 10"
        ]

        output = self.ParseAndSearch(data, UCS())
        self.AssertLists(output, expect)

    def test_Astar_Test4(self):
        data = [
            "A*",
            "5 5 5",
            "1 0 4",
            "3 1 2",
            "12",
            "0 0 1 1",
            "0 3 3 8",
            "1 0 1 2",
            "1 0 4 3 16",
            "1 1 3 1 3 5 12 16 17",
            "1 1 4 4 6 12",
            "1 2 2 17",
            "1 2 3 4 9",
            "2 1 2 1 12 13",
            "2 1 3 2 13",
            "3 1 1 13 5",
            "3 1 2 2 6"
        ]

        expect = [
            "38",
            "4",
            "1 0 4 0",
            "1 1 3 14",
            "2 1 2 14",
            "3 1 2 10"
        ]

        output = self.ParseAndSearch(data, UCS())
        self.AssertLists(output, expect)