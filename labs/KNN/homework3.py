import heapq


class Utilities:
    def ConvertToInt(str):
        ret = []
        for s in str.split():
            ret.append(int(s))
        return ret

    def OutputToFile(path, data, delimiter):
        isFirst = True
        with open(path, "w") as file:
            for line in data:
                if not isFirst:
                    file.write(delimiter)
                isFirst = False
                file.writelines(line)

    def MoveOfAction(actionCode):
        coord = None
        if actionCode == 1:
            coord = [1, 0, 0]
        if actionCode == 2:
            coord = [-1, 0, 0]
        if actionCode == 3:
            coord = [0, 1, 0]
        if actionCode == 4:
            coord = [0, -1, 0]
        if actionCode == 5:
            coord = [0, 0, 1]
        if actionCode == 6:
            coord = [0, 0, -1]
        if actionCode == 7:
            coord = [1, 1, 0]
        if actionCode == 8:
            coord = [1, -1, 0]
        if actionCode == 9:
            coord = [-1, 1, 0]
        if actionCode == 10:
            coord = [-1, -1, 0]
        if actionCode == 11:
            coord = [1, 0, 1]
        if actionCode == 12:
            coord = [1, 0, -1]
        if actionCode == 13:
            coord = [-1, 0, 1]
        if actionCode == 14:
            coord = [-1, 0, -1]
        if actionCode == 15:
            coord = [0, 1, 1]
        if actionCode == 16:
            coord = [0, 1, -1]
        if actionCode == 17:
            coord = [0, -1, 1]
        if actionCode == 18:
            coord = [0, -1, -1]

        return coord

    def ParseActions(data):
        grid = {}

        for s in data:
            if not grid.get(s[0], {}):
                grid[s[0]] = {}

            if not grid[s[0]].get(s[1], {}):
                grid[s[0]][s[1]] = {}

            grid[s[0]][s[1]][s[2]] = s[3:]

        return grid


class Node:
    cost = 0
    _path = []

    def __init__(self, myself, cost, paths):
        self.cost += cost
        self._path = paths[:]  # TODO: reduce
        self.cost += sum([p[3] for p in paths])
        self._path.append(myself + [cost])

    def MyCoord(self):
        return self._path[-1][:-1]

    def Output(self):
        ans = [str(self.cost), str(len(self._path))]
        [ans.append("{0} {1} {2} {3}".format(p[0], p[1], p[2], p[3])) for p in self._path]
        return ans

    def GetNewCoordAfterAction(self, actionCode):
        newCoord = Utilities.MoveOfAction(actionCode)
        original = self._path[-1]
        newCoord[0] += original[0]
        newCoord[1] += original[1]
        newCoord[2] += original[2]
        return newCoord

    def GenerateNext(self, args):
        return Node(self.GetNewCoordAfterAction(args[0]), args[1], self._path)

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.MyCoord() == other.MyCoord()


class BFS:
    def CostOfAction(self, args):
        return 1

    def CreateNode(self, myself, cost, paths):
        return Node(myself, cost, paths)

    def GenerateNextNode(self, cur, args):
        return cur.GenerateNext(args)

    def Evaluate(self, start, end, actions):

        pq = []
        outPq = []

        heapq.heappush(pq, self.CreateNode(start, 0, []))

        while pq:
            cur = heapq.heappop(pq)

            if cur in outPq:
                continue

            outPq.append(cur)
            myCoor = cur.MyCoord()

            if myCoor == end:
                return cur.Output()

            for i in actions.get(myCoor[0], {}).get(myCoor[1], {}).get(myCoor[2], []):

                next = self.GenerateNextNode(cur, [i, self.CostOfAction([i])])

                if next in pq:
                    inPq = pq[pq.index(next)]
                    if next.cost < inPq.cost:
                        inPq.cost = next.cost
                        heapq.heapify(pq)
                    continue

                heapq.heappush(pq, next)

        return ["FAIL"]


class UCS(BFS):

    def CostOfAction(self, args):
        if args[0] <= 6:
            return 10
        return 14


class AstarNode(Node):
    _eta = 0

    def __init__(self, myself, cost, paths, end):
        super().__init__(myself, cost, paths)
        coord = self.MyCoord()
        self._eta = int((abs(coord[0] - end[0]) ** 3 + abs(coord[1] - end[1]) ** 3 +
            abs(coord[2] - end[2]) ** 3) ** (1 / 3))

    def GenerateNext(self, args):
        return AstarNode(self.GetNewCoordAfterAction(args[0]), args[1], self._path, args[2])

    def __lt__(self, other):
        return self.cost + self._eta < other.cost + other._eta


class Astar(UCS):
    _end = []

    def __init__(self, end):
        self._end = end[:]

    def CreateNode(self, myself, cost, paths):
        return AstarNode(myself, cost, paths, self._end)

    def GenerateNextNode(self, cur, args):
        return cur.GenerateNext(args + [self._end])


if __name__ == '__main__':

    option = None
    dimensions = []
    start = []
    end = []
    data = []

    search = None

    with open("input.txt", "r") as file:
        option = file.readline()[0:-1]
        dimensions = Utilities.ConvertToInt(file.readline())
        start = Utilities.ConvertToInt(file.readline())
        end = Utilities.ConvertToInt(file.readline())
        data = [Utilities.ConvertToInt(line) for line in list(file)[1:]]

    if option == "A*":
        search = Astar(end)

    if option == "BFS":
        search = BFS()

    if option == "UCS":
        search = UCS()

    Utilities.OutputToFile("output.txt", search.Evaluate(start, end, Utilities.ParseActions(data)), "\n")
