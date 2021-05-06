from abc import abstractmethod
from copy import deepcopy
import numpy as np

class Vertex:
    def __init__(self):
        self.matrixUp = []
        self.matrixDown = []

    @abstractmethod
    def up(self):
        pass

    @abstractmethod
    def down(self):
        pass

    def readMatrixDown(self):
        for i in range(len(self.matrixUp)):
            self.matrixDown[i] = list(map(float, input().split()))
        self.matrixDown = np.array(self.matrixDown)

    def buildMatrixDown(self):
        r = len(self.matrixUp)
        c = len(self.matrixUp[0])
        self.matrixDown = np.zeros(shape=(r, c))

class Var(Vertex):
    def __init__(self, r, c):
        super().__init__()
        self.r = r
        self.c = c

    def readMatrixDown(self):
        super().readMatrixDown()

    def buildMatrixDown(self):
        super().buildMatrixDown()

    def readMatrixUp(self):
        for i in range(self.r):
            self.matrixUp.append(list(map(float, input().split())))
        self.matrixUp = np.array(self.matrixUp)
        self.buildMatrixDown()

    def up(self):
        self.buildMatrixDown()

    def down(self):
        if len(self.matrixDown) == 0:
            self.buildMatrixDown()
        pass

class Tnh(Vertex):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def readMatrixDown(self):
        super().readMatrixDown()

    def buildMatrixDown(self):
        super().buildMatrixDown()

    def up(self):
        self.matrixUp = np.array(deepcopy(vertices[self.x].matrixUp))
        self.matrixUp = np.tanh(self.matrixUp)
        self.buildMatrixDown()

    def down(self):
        for i in range(len(self.matrixDown)):
            for j in range(len(self.matrixDown[0])):
                vertices[self.x].matrixDown[i][j] += self.matrixDown[i][j] * (1 - self.matrixUp[i][j] ** 2)


class Rlu(Vertex):
    def __init__(self, alpha, x):
        super().__init__()
        self.alpha = alpha
        self.x = x

    def readMatrixDown(self):
        super().readMatrixDown()

    def buildMatrixDown(self):
        super().buildMatrixDown()

    def up(self):
        matrix = np.array(deepcopy(vertices[self.x].matrixUp))
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] < 0:
                    matrix[i][j] /= self.alpha
        self.matrixUp = matrix
        self.buildMatrixDown()

    def down(self):
        for i in range(len(self.matrixDown)):
            for j in range(len(self.matrixDown[0])):
                if vertices[self.x].matrixUp[i][j] >= 0:
                    vertices[self.x].matrixDown[i][j] += self.matrixDown[i][j]
                else:
                    vertices[self.x].matrixDown[i][j] += self.matrixDown[i][j] / self.alpha


class Mul(Vertex):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def readMatrixDown(self):
        super().readMatrixDown()

    def buildMatrixDown(self):
        super().buildMatrixDown()

    def up(self):
        matrix_a = np.array(deepcopy(vertices[self.a].matrixUp))
        matrix_b = np.array(deepcopy(vertices[self.b].matrixUp))
        self.matrixUp = deepcopy(np.dot(matrix_a, matrix_b))
        self.buildMatrixDown()

    def down(self):
        for i in range(len(vertices[self.a].matrixDown)):
            for j in range(len(vertices[self.b].matrixDown)):
                for p in range(len(vertices[self.b].matrixDown[0])):
                    vertices[self.a].matrixDown[i][j] += vertices[self.b].matrixUp[j][p] * self.matrixDown[i][p]

        for i in range(len(vertices[self.b].matrixDown)):
            for j in range(len(vertices[self.b].matrixDown[0])):
                for p in range(len(vertices[self.a].matrixDown)):
                    vertices[self.b].matrixDown[i][j] += vertices[self.a].matrixUp[p][i] * self.matrixDown[p][j]


class Sum(Vertex):
    def __init__(self, length, us):
        super().__init__()
        self.length = length
        self.us = us

    def readMatrixDown(self):
        super().readMatrixDown()

    def buildMatrixDown(self):
        super().buildMatrixDown()

    def up(self):
        matrix_sum = np.array(deepcopy(vertices[self.us[0]].matrixUp))
        for i in range(1, self.length):
            matrix_sum = matrix_sum + np.array(deepcopy(vertices[self.us[i]].matrixUp))
        self.matrixUp = np.array(deepcopy(matrix_sum))
        self.buildMatrixDown()

    def down(self):
        for i in range(self.length):
            for j in range(len(self.matrixDown)):
                for p in range(len(self.matrixDown[0])):
                    vertices[self.us[i]].matrixDown[j][p] += self.matrixDown[j][p]

class Had(Vertex):
    def __init__(self, length, us):
        super().__init__()
        self.length = length
        self.us = us

    def readMatrixDown(self):
        super().readMatrixDown()

    def buildMatrixDown(self):
        super().buildMatrixDown()

    def up(self):
        matrix = np.array(deepcopy(vertices[self.us[0]].matrixUp))
        for i in range(1, self.length):
            matrix = matrix * np.array(deepcopy(vertices[self.us[i]].matrixUp))

        self.matrixUp = matrix
        self.buildMatrixDown()

    def down(self):
        for i in range(len(self.matrixUp)):
            for j in range(len(self.matrixUp[0])):
                for p in range(self.length):
                    coef = 1.0
                    for d in range(self.length):
                        if p != d:
                            coef *= vertices[self.us[d]].matrixUp[i][j]
                    vertices[self.us[p]].matrixDown[i][j] += coef * self.matrixDown[i][j]

n, m, k = map(int, input().split())

vertices = []

for i in range(n):
    query = list(map(str, input().split()))
    if query[0] == 'var':
        r, c = int(query[1]), int(query[2])
        vertices.append(Var(r, c))
    elif query[0] == 'tnh':
        x = int(query[1]) - 1
        vertices.append(Tnh(x))
    elif query[0] == 'rlu':
        alpha, x = int(query[1]), int(query[2]) - 1
        vertices.append(Rlu(alpha, x))
    elif query[0] == 'mul':
        a, b = int(query[1]) - 1, int(query[2]) - 1
        vertices.append(Mul(a, b))
    elif query[0] == 'sum':
        length = int(query[1])
        us = [int(query[j]) - 1 for j in range(2, len(query))]
        vertices.append(Sum(length, us))
    elif query[0] == 'had':
        length = int(query[1])
        us = [int(query[j]) - 1 for j in range(2, len(query))]
        vertices.append(Had(length, us))

for i in range(m):
    vertices[i].readMatrixUp()

for i in range(m, n):
    vertices[i].up()

for i in range(k):
    for row in vertices[n - k + i].matrixUp:
        print(' '.join([str(elem) for elem in row]))
    vertices[n - k + i].readMatrixDown()

for i in range(n - 1, -1, -1):
    vertices[i].down()

for i in range(m):
    for row in vertices[i].matrixDown:
        print(' '.join([str(elem) for elem in row]))
 
