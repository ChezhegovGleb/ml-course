import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

fileName = "2.txt"
h = 0.2
countIterate = 2000
tao = 0.001
temp = 0.01

coef = 5

lossVector = []
iter = []
minmax = []
dataTest = []

def dataTrain_minmax(dataTrain, m, n):
    minmax = [[0.0, 0.0] for i in range(m + 2)]
    for j in range(1, m + 1):
        valueMin = dataTrain[0][j]
        valueMax = dataTrain[0][j]
        for i in range(n):
            valueMin = min(valueMin, dataTrain[i][j])
            valueMax = max(valueMax, dataTrain[i][j])
        minmax[j] = [valueMin, valueMax]
    return minmax

def normalize(dataTrain, minmax):
    for row in dataTrain:
        for i in range(1, len(row) - 1):
            if (minmax[i][1] - minmax[i][0] == 0):
                row[i] = 0
            else:
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def denormalize(oldWeights, minmax, m):
    for i in range(1, m + 1):
        if (minmax[i][1] - minmax[i][0] != 0):
            oldWeights[0] -= oldWeights[i] / (minmax[i][1] - minmax[i][0]) * minmax[i][0]
            oldWeights[i] /= (minmax[i][1] - minmax[i][0])
        else:
            oldWeights[i] = 0

def sgd():
    weights = [random.uniform(-1 / (2 * m), 1 / (2 * m)) for i in range(m + 1)]

    evalFunc = 0.0

    for it in trange(3, countIterate + 3):
        index = int(random.uniform(0, n + 1)) % n
        realY = dataTrain[index][m + 1]

        grad = [0 for i in range(m + 1)]

        curY = 0.0
        for j in range(m + 1):
            curY += weights[j] * dataTrain[index][j]

        diffY = curY - realY
        stepReducer =  np.log2(it)
        curH = h / stepReducer

        for i in range(m + 1):
            grad[i] = dataTrain[index][i] * diffY
            weights[i] *= (1 - tao * curH)
            weights[i] -= curH * grad[i]

        nrmse = 0.0

        for ind in range(len(dataTest)):
            ans = 0.0
            xTest = dataTest[ind]
            for j in range(m + 1):
                ans += xTest[j] * weights[j]
            realAns = xTest[-1]
            curLoss = (ans - realAns) ** 2
            nrmse += curLoss

        nrmse /= len(dataTest)
        nrmse = math.sqrt(nrmse)
        nrmse /= diffTestY

        lossVector.append(nrmse)

    return weights

def svd(x, y, coef):
    xTranspose = np.transpose(x)
    return np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, x) + coef * np.identity(len(xTranspose))), xTranspose), y)

if __name__ == '__main__':
    with open(fileName) as fileName:
        m = int(next(fileName))
        n = int(next(fileName))

        dataTrain = [[1] for i in range(n)]

        x = []
        y = []

        for i in range(n):
            dataTrain[i] += list(map(int, next(fileName).split()))
            x.append(dataTrain[i][:-1])
            y.append(dataTrain[i][-1])

        k = int(next(fileName))

        dataTest = [[1] for i in range(k)]

        for i in range(k):
            dataTest[i] += list(map(int, next(fileName).split()))

        minmax = dataTrain_minmax(dataTrain, m, n)
        normalize(dataTrain, minmax)

        normalize(dataTest, minmax)

        yMax = dataTrain[0][-1]
        yMin = dataTrain[0][-1]

        for i in range(n):
            yMin = min(yMin, dataTrain[i][-1])
            yMax = max(yMax, dataTrain[i][-1])

        diffTrainY = yMax - yMin

        yMax = dataTest[0][-1]
        yMin = dataTest[0][-1]

        for i in range(k):
            yMin = min(yMin, dataTest[i][-1])
            yMax = max(yMax, dataTest[i][-1])

        diffTestY = yMax - yMin

        weights = sgd()

        nrmseTrain = 0.0

        for ind in range(len(dataTrain)):
            ans = 0.0
            xTest = dataTrain[ind]
            for j in range(m + 1):
                ans += xTest[j] * weights[j]
            realAns = xTest[-1]
            curLoss = (ans - realAns) ** 2
            nrmseTrain += curLoss

        nrmseTrain /= len(dataTrain)
        nrmseTrain = math.sqrt(nrmseTrain)
        nrmseTrain /= diffTrainY

        nrmseTest = 0.0

        for ind in range(len(dataTest)):
            ans = 0.0
            xTest = dataTest[ind]
            for j in range(m + 1):
                ans += xTest[j] * weights[j]
            realAns = xTest[-1]
            curLoss = (ans - realAns) ** 2
            nrmseTest += curLoss

        nrmseTest /= len(dataTest)
        nrmseTest = math.sqrt(nrmseTest)
        nrmseTest /= diffTestY

        print('final nrmse on train: {}, final nrmse on test {}'.format(nrmseTrain, nrmseTest))

        plt.plot(lossVector)
        plt.xlabel("Iteration number")
        plt.ylabel("NRMSE")
        plt.title("Graphic")
        plt.show()

        weights = svd(x, y, coef)

        nrmseTest = 0.0

        for ind in range(len(dataTest)):
            ans = 0.0
            xTest = dataTest[ind]
            for j in range(m + 1):
                ans += xTest[j] * weights[j]
            realAns = xTest[-1]
            curLoss = (ans - realAns) ** 2
            nrmseTest += curLoss

        nrmseTest /= len(dataTest)
        nrmseTest = math.sqrt(nrmseTest)
        nrmseTest /= diffTestY
        nrmseTest /= 100

        print('final svd on test {}'.format(nrmseTest))