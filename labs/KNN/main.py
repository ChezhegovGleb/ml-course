from math import cos
from math import e
from math import pi

import pandas as pd
import copy
import matplotlib.pyplot as plt

def calculateDist(obj1, obj2, metric):
    dist = 0.0

    if metric == "manhattan":
        for i in range(len(obj1)):
            dist += abs(obj1[i] - obj2[i])

    if metric == "euclidean":
        for i in range(len(obj1)):
            dist += (obj1[i] - obj2[i]) ** 2
        dist = dist ** 0.5

    if metric == "chebyshev":
        for i in range(len(obj1)):
            dist = max(dist, abs(obj1[i] - obj2[i]))

    return dist

def kernelSmoothing(u, kernel):
    if (kernel == "uniform"):
        return 0.5 if (abs(u) < 1) else 0

    if (kernel == "triangular"):
        return (1 - abs(u)) if (abs(u) < 1) else 0

    if (kernel == "epanechnikov"):
        return 0.75 * (1 - u ** 2) if (abs(u) < 1) else 0

    if (kernel == "quartic"):
        return (15.0 / 16.0 * (1 - u ** 2) ** 2) if (abs(u) < 1) else 0

    if (kernel == "triweight"):
        return (35.0 / 32.0 * (1 - u ** 2) ** 3) if (abs(u) < 1) else 0

    if (kernel == "tricube"):
        return (70.0 / 81.0 * (1 - abs(u) ** 3) ** 3) if (abs(u) < 1) else 0

    if (kernel == "gaussian"):
        return (1 / (2 * pi) ** 0.5 * e ** (-0.5 * u ** 2))

    if (kernel == "cosine"):
        return (pi / 4 * cos(pi * u / 2)) if (abs(u) < 1) else 0

    if (kernel == "logistic"):
        return 1 / (e ** u + 2 + e ** (-u))

    if (kernel == "sigmoid"):
        return 2 / (pi * (e ** u + e ** (-u)))

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
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def getFscore(confMatrix, lenData):
    f = 0

    rowSum = [0] * classes
    colSum = [0] * classes

    for i in range(classes):
        for j in range(classes):
            rowSum[i] += confMatrix[i][j]
            colSum[j] += confMatrix[i][j]

    for i in range(classes):
        prec = 0
        rec = 0
        if (rowSum[i] != 0):
            prec = confMatrix[i][i] / rowSum[i]
        if (colSum[i] != 0):
            rec = confMatrix[i][i] / colSum[i]
        if (prec * rec != 0):
            f += 2.0 * prec * rec * rowSum[i] / (prec + rec)

    f /= lenData
    return f

def oneHot(data_values, window, paramWindow, metric, kernel):
    listData = copy.deepcopy(data_values.tolist())
    arrData = copy.deepcopy(listData)

    for i in range(len(arrData)):
        arrData[i].pop()

    matrix = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]

    confMatrix = [[0] * classes for i in range(classes)]
    for it in range(n):
        result = []
        targetClass = listData[it][-1]
        for cl in range(classes):

            data = copy.deepcopy(arrData)
            target = data[it]

            for i in range(len(data)):
                if (listData[i][-1] == 0):
                    data[i].append(matrix[cl][0])
                if (listData[i][-1] == 1):
                    data[i].append(matrix[cl][1])
                if (listData[i][-1] == 2):
                    data[i].append(matrix[cl][2])

                data[i].append(calculateDist(data[i][:-1], target, metric))

            del (data[it])
            data.sort(key=lambda obj: obj[-1])

            sumUp = 0.0
            sumDown = 0.0

            if window == "fixed":
                if (paramWindow != 0):
                    for obj in data:
                        newDist = kernelSmoothing(obj[-1] / paramWindow, kernel)
                        sumUp += obj[-2] * newDist
                        sumDown += newDist

            if window == "variable":
                if (data[paramWindow][-1] != 0):
                    for obj in data:
                        newDist = kernelSmoothing(obj[-1] / data[paramWindow][-1], kernel)
                        sumUp += obj[-2] * newDist
                        sumDown += newDist

            if (sumDown != 0):
                resultClass = sumUp / sumDown
            else:
                meanSum = 0.0
                cnt = 0
                for obj in data:
                    if (obj[:-2] == target):
                        cnt += 1
                        meanSum += obj[-2]
                if cnt != 0:
                    resultClass = meanSum / cnt
                else:
                    for obj in data:
                        meanSum += obj[-2]
                    resultClass = meanSum / len(data)

            result.append(resultClass)

        bestClass = 0
        bestValue = max(result)
        for c in range(classes):
            if (bestValue == result[c]):
                bestClass = c

        confMatrix[int(targetClass)][bestClass] += 1

    f = getFscore(confMatrix, n)

    # best - 0.7971403722201207 manhattan tricube variable 13
    # best - 0.7993555565188492 manhattan tricube fixed 0.4

    # print(f)
    return f

def naive(data_values, window, paramWindow, metric, kernel):
    confMatrix = [[0] * classes for i in range(classes)]
    for it in range(n):

        data = data_values.tolist()
        target = data[it][:-1]
        targetClass = data[it][-1]
        del (data[it])

        for i in range(len(data)):
            data[i].append(calculateDist(data[i][:-1], target, metric))

        data.sort(key=lambda obj: obj[-1])

        sumUp = 0.0
        sumDown = 0.0

        if window == "fixed":
            if (paramWindow != 0):
                for obj in data:
                    newDist = kernelSmoothing(obj[-1] / paramWindow, kernel)
                    sumUp += obj[-2] * newDist
                    sumDown += newDist

        if window == "variable":
            if (data[paramWindow][-1] != 0):
                for obj in data:
                    newDist = kernelSmoothing(obj[-1] / data[paramWindow][-1], kernel)
                    sumUp += obj[-2] * newDist
                    sumDown += newDist

        resultClass = 0

        if (sumDown != 0):
            resultClass = round(sumUp / sumDown)
        else:
            meanSum = 0.0
            cnt = 0
            for obj in data:
                if (obj[:-2] == target):
                    cnt += 1
                    meanSum += obj[-2]
            if cnt != 0:
                resultClass = round(meanSum / cnt)
            else:
                for obj in data:
                    meanSum += obj[-2]
                resultClass = round(meanSum / len(data))

        confMatrix[int(targetClass)][int(resultClass)] += 1

    f = getFscore(confMatrix, n)

    return f

    # best - 0.7771334543193761 manhattan tricube variable 6
    # best - 0.7481577558908808 manhattan triweight fixed 0.4


filename = "cars1.csv"

dataset = pd.read_csv(filename)

data_values = dataset.values
minmax = dataset_minmax(data_values)
normalize(data_values, minmax)

#print(data_values)

data = data_values.tolist()

n = len(data_values)
n -= 1

classes = 3

# Naive method

f = 0
metric = "manhattan"
kernel = "tricube"
window = "fixed"
paramWindow = 0.4

naive(data_values, "fixed", 0.4, metric, kernel)
naive(data_values, "variable", 6, metric, kernel)

# OneHot

oneHot(data_values, "variable", 13, metric, kernel)
oneHot(data_values, "fixed", 0.4, metric, kernel)

# draw variable graphics

x = [x for x in range(0, 300, 20)]
y = [oneHot(data_values, "variable", y) for y in range(0, 300, 20)]

plt.plot(x, y)
plt.xlabel("param_window")
plt.ylabel("f_score")
plt.title("Variable window")
plt.show()

# draw fixed graphics

x = []
y = []
left = 0
right = 3
step = 0.2
while (left < right):
    x.append(left)
    y.append(oneHot(data_values, "fixed", left, metric, kernel))
    left += step

plt.plot(x, y)
plt.xlabel("param window")
plt.ylabel("f_score")
plt.title("Fixed window")
plt.show()

# Search best params
# # Naive method

#
# metric = "euclidean"
# kernel = "quartic"
# window = "variable"
#
# metrics = ["manhattan", "euclidean", "chebyshev"]
# kernels = ["uniform", "triangular", "epanechnikov", "quartic", "triweight", "tricube", "gaussian", "cosine", "logistic", "sigmoid"]
# windows = ["fixed", "variable"]
#
# f_max = 0
# max_metric = 0
# max_kernel = 0
# max_window = 0
# max_paramWindow = 0
#
# paramsWindow = [5, 6, 7, 8, 9]
#
# for window in windows:
#     for paramWindow in range(1, 200, 10):
#         print(paramWindow)
#         for kernel in kernels:
#             for metric in metrics:
#                 confMatrix = [[0] * classes for i in range(classes)]
#                 for it in range(n):
#
#                     data = data_values.tolist()
#                     target = data[it][:-1]
#                     targetClass = data[it][-1]
#                     del(data[it])
#
#                     for i in range(len(data)):
#                         data[i].append(calculateDist(data[i][:-1], target, metric))
#
#                     data.sort(key=lambda obj: obj[-1])
#
#                     sumUp = 0.0
#                     sumDown = 0.0
#
#                     if window == "fixed":
#                         if (paramWindow != 0):
#                             for obj in data:
#                                 newDist = kernelSmoothing(obj[-1] / paramWindow, kernel)
#                                 sumUp += obj[-2] * newDist
#                                 sumDown += newDist
#
#                     if window == "variable":
#                         if (data[paramWindow][-1] != 0):
#                             for obj in data:
#                                 newDist = kernelSmoothing(obj[-1] / data[paramWindow][-1], kernel)
#                                 sumUp += obj[-2] * newDist
#                                 sumDown += newDist
#
#                     resultClass = 0
#
#                     if (sumDown != 0):
#                         resultClass = round(sumUp / sumDown)
#                     else:
#                         meanSum = 0.0
#                         cnt = 0
#                         for obj in data:
#                             if (obj[:-2] == target):
#                                 cnt += 1
#                                 meanSum += obj[-2]
#                         if cnt != 0:
#                             resultClass = round(meanSum / cnt)
#                         else:
#                             for obj in data:
#                                 meanSum += obj[-2]
#                             resultClass = round(meanSum / len(data))
#
#                     confMatrix[int(targetClass)][int(resultClass)] += 1
#
#                 f = getFscore(confMatrix, 391)
#                 if (f > f_max):
#                     f_max = f
#                     max_metric = metric
#                     max_kernel = kernel
#                     max_window = window
#                     max_paramWindow = paramWindow
#                     print(f_max, max_metric, max_kernel, max_window, max_paramWindow)
#
#
# #best - 0.7771334543193761 manhattan tricube variable 6
# #best - 0.6543888942415784 manhattan triweight fixed 1
#
# # print(f_max, max_metric, max_kernel, max_window, max_paramWindow)
#
# # OneHot
#
# listData = copy.deepcopy(data_values.tolist())
# arrData = copy.deepcopy(listData)
#
# for i in range(len(arrData)):
#     arrData[i].pop()
#
# matrix = [[1, 0, 0],
#           [0, 1, 0],
#           [0, 0, 1]]
#
# f_max = 0
# max_metric = 0
# max_kernel = 0
# max_window = 0
# max_paramWindow = 0
#
# paramsWindow = [2, 3, 4]
#
# for window in windows:
#     for paramWindow in paramsWindow:
#         print(paramWindow)
#         for kernel in kernels:
#             for metric in metrics:
#                 confMatrix = [[0] * classes for i in range(classes)]
#                 for it in range(n):
#                     result = []
#                     targetClass = listData[it][-1]
#                     for cl in range(classes):
#
#                         data = copy.deepcopy(arrData)
#                         target = data[it]
#
#                         for i in range(len(data)):
#                             if (listData[i][-1] == 0):
#                                 data[i].append(matrix[cl][0])
#                             if (listData[i][-1] == 1):
#                                 data[i].append(matrix[cl][1])
#                             if (listData[i][-1] == 2):
#                                 data[i].append(matrix[cl][2])
#
#                             data[i].append(calculateDist(data[i][:-1], target, metric))
#
#                         del (data[it])
#                         data.sort(key=lambda obj: obj[-1])
#
#                         sumUp = 0.0
#                         sumDown = 0.0
#
#                         if window == "fixed":
#                             if (paramWindow != 0):
#                                 for obj in data:
#                                     newDist = kernelSmoothing(obj[-1] / paramWindow, kernel)
#                                     sumUp += obj[-2] * newDist
#                                     sumDown += newDist
#
#                         if window == "variable":
#                             if (data[paramWindow][-1] != 0):
#                                 for obj in data:
#                                     newDist = kernelSmoothing(obj[-1] / data[paramWindow][-1], kernel)
#                                     sumUp += obj[-2] * newDist
#                                     sumDown += newDist
#
#                         resultClass = 0
#
#                         if (sumDown != 0):
#                             resultClass = sumUp / sumDown
#                         else:
#                             meanSum = 0.0
#                             cnt = 0
#                             for obj in data:
#                                 if (obj[:-2] == target):
#                                     cnt += 1
#                                     meanSum += obj[-2]
#                             if cnt != 0:
#                                 resultClass = meanSum / cnt
#                             else:
#                                 for obj in data:
#                                     meanSum += obj[-2]
#                                 resultClass = meanSum / len(data)
#
#                         result.append(resultClass)
#
#                     bestClass = 0
#                     bestValue = max(result)
#                     for c in range(classes):
#                         if (bestValue == result[c]):
#                             bestClass = c
#
#                     confMatrix[int(targetClass)][bestClass] += 1
#
#                 print(confMatrix)
#
#                 f = getFscore(confMatrix, 391)
#
#                 if (f > f_max):
#                     f_max = f
#                     max_metric = metric
#                     max_kernel = kernel
#                     max_window = window
#                     max_paramWindow = paramWindow
#                     print(f_max, max_metric, max_kernel, max_window, max_paramWindow)
#
#
# #best - 0.7971403722201207 manhattan tricube variable 13
# #best - 0.7493606138107417 manhattan tricube fixed 1
# print(f_max, max_metric, max_kernel, max_window, max_paramWindow)
