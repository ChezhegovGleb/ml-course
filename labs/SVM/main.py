import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def dataset_minmax(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        if (i == len(dataset[0]) - 1):
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax

def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if (i == len(row) - 1):
                if (row[i] == 'P'):
                    row[i] = -1
                else:
                    row[i] = 1
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def k_linear(x, y):
    res = 0.0
    for i in range(len(x)):
        res += x[i] * y[i]
    return res

def k_polinom(x, y, p):
    return pow(k_linear(x, y), p)

def k_gaus(x, y, b):
    return np.exp(-b * pow(np.linalg.norm(x - y), 2))

def getKernel(type, x, y):
    if (type[0] == "l"):
        return k_linear(x, y)
    elif (type[0] == "p"):
        return k_polinom(x, y, type[1])
    else:
        return k_gaus(x, y, type[1])

def sign(x):
    if (x == 0):
        return 0.0
    elif (x > 0):
        return 1.0
    else:
        return -1.0

def drawGraphic(objects, answers, kernel, C):
    alphas, threshold = svm_smo(objects, answers, kernel, C)
    x_i = 0.0
    step = 0.005

    plt.title("Graphic")
    plt.xlabel("X")
    plt.ylabel("Y")

    while (x_i < 1.01):
        y_i = 0.0
        while (y_i < 1.01):
            res = sign(getSumFunc(x, y, alphas, [x_i, y_i], kernel, threshold))
            if res <= 0:
                plt.plot(x_i, y_i, color="#ccf5fc", marker="s")
            else:
                plt.plot(x_i, y_i, color="#ffd6d4", marker="s")
            y_i += step
        x_i += step

    for i in range(len(objects)):
        if answers[i] == 1:
            plt.plot(objects[i][0], objects[i][1], 'ro')
        else:
            plt.plot(objects[i][0], objects[i][1], 'bo')

    plt.show()


def svm_smo(objects, answers, kernelType, C):
    threshold = 0
    iter = 0
    passes = 0
    alphas = [0.0] * len(objects)

    while iter < iters and passes < max_passes:
        changed_alphas = 0
        for i in range(len(objects)):
            E1 = getSumFunc(objects, answers, alphas, objects[i], kernelType, threshold) - answers[i]
            if (answers[i] * E1 < -tol and alphas[i] < C) or (answers[i] * E1 > tol and alphas[i] > 0):
                j = random.randint(0, len(objects) - 1)
                while (j == i):
                    j = random.randint(0, len(objects) - 1)

                E2 = getSumFunc(objects, answers, alphas, objects[j], kernelType, threshold) - answers[j]
                H = C

                if answers[j] != answers[i]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(H, alphas[j] - alphas[i] + C)
                else:
                    L = max(0.0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue

                k_11 = getKernel(kernelType, objects[i], objects[i])
                k_12 = getKernel(kernelType, objects[i], objects[j])
                k_22 = getKernel(kernelType, objects[j], objects[j])

                ita = 2 * k_12 - k_11 - k_22
                if ita >= 0:
                    continue

                oldA1 = alphas[i]
                oldA2 = alphas[j]
                alphas[j] = alphas[j] - (E1 - E2) * answers[j] / ita

                if alphas[j] <= H:
                    if alphas[j] < L:
                        alphas[j] = L
                else:
                    alphas[j] = H

                if abs(alphas[j] - oldA2) < 1e-5:
                    continue

                alphas[i] += answers[i] * answers[j] * (oldA2 - alphas[j])
                threshold1 = threshold - E1 - answers[i] * (alphas[i] - oldA1) * k_11 - answers[j] * (alphas[j] - oldA2) * k_12
                threshold2 = threshold - E2 - answers[i] * (alphas[i] - oldA1) * k_12 - answers[j] * (alphas[j] - oldA2) * k_22

                if 0 < alphas[i] < C:
                    threshold = threshold1
                else:
                    if 0 < alphas[j] < C:
                        threshold = threshold2
                    else:
                        threshold = (threshold1 + threshold2) / 2.0
                changed_alphas += 1
        if changed_alphas == 0:
            iter += 1
        else:
            iter = 0
        passes += 1
    return alphas, threshold

def getSumFunc(objects, answers, alphas, x2, kernelType, threshold):
    sumFunc = threshold
    for i in range(len(objects)):
        sumFunc += answers[i] * alphas[i] * getKernel(kernelType, objects[i], x2)
    return sumFunc

def crossVal(objects, answers, k, kernel, C):
    features = objects
    blockSize = int(len(features) / k)
    accuracy = 0.0
    for i in range(k):
        yTest = answers[i * blockSize:(i + 1) * blockSize]
        yTrain = answers[0:i * blockSize] + answers[(i + 1) * blockSize:]
        xTest = features[(i * blockSize):((i + 1) * blockSize)]
        xTrain = features[0:(i * blockSize)] + features[(i + 1) * blockSize:]
        alphas, threshold = svm_smo(xTrain, yTrain, kernel, C)
        success = 0
        for j in range(len(xTest)):
            if yTest[j] == sign(getSumFunc(xTrain, yTrain, alphas, xTest[j], kernel, threshold)):
                success += 1
        accuracy += success / (len(xTest))
    return accuracy / k

max_passes = 100
tol = 1e-3
iters = 20

filename = "chips.csv"
filename2 = "geyser.csv"

dataset = pd.read_csv(filename2)
data_values = dataset.values
minmax = dataset_minmax(data_values)
normalize(data_values, minmax)

data_values = shuffle(data_values)

x = []
y = []

for i in range(len(data_values)):
    x.append(data_values[i][:-1])
    y.append(data_values[i][-1])

# chips

drawGraphic(x, y, ["l", None], 1.0) # 0.4027272727272727
drawGraphic(x, y, ["p", 5], 100.0) # 0.6636363636363637
drawGraphic(x, y, ["g", 5], 10.0) # 0.8454545454545455

# geyser

drawGraphic(x, y, ["l", None], 0.1) # 0.8954545454545455
drawGraphic(x, y, ["p", 2], 5) # 0.9045454545454547
drawGraphic(x, y, ["g", 1], 10) # 0.9045454545454545


bestAccuracy = 0

for beta in [1, 2, 3, 4, 5]:
    kernel = ["g", beta]
    for C in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        result = crossVal(x, y, 7, kernel, C)
        if result >= bestAccuracy:
            bestAccuracy = result
            print(kernel[0] + " " + str(kernel[1]) + " " + str(C) + " " + str(bestAccuracy))

for p in [2, 3, 4, 5]:
    kernel = ["p", p]
    for C in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        result = crossVal(x, y, 20, kernel, C)
        if result >= bestAccuracy:
            bestAccuracy = result
            print(kernel[0] + " " + str(kernel[1]) + " " + str(C) + " " + str(bestAccuracy))

for l in [1, 2, 3, 4, 5]:
    kernel = ["l", None]
    for C in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        result = crossVal(x, y, 15, kernel, C)
        if result >= bestAccuracy:
            bestAccuracy = result
            print(kernel[0] + " " + str(kernel[1]) + " " + str(C) + " " + str(bestAccuracy))