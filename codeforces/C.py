from math import pi
from math import e
from math import cos

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


def main():

    n, m = map(int, input().split())

    dataset = []
    for i in range(n):
        dataset.append(list(map(int, input().split())))

    q = list(map(int, input().split()))
    metric = input()
    kernel = input()
    window = input()

    for i in range(len(dataset)):
        dataset[i].append(calculateDist(dataset[i][:-1], q, metric))

    dataset.sort(key=lambda obj: obj[-1])

    if window == "fixed":
        h = float(input())

        sumUp = 0.0
        sumDown = 0.0

        if (h != 0):
            for obj in dataset:
                    newDist = kernelSmoothing(obj[-1] / h, kernel)
                    sumUp += obj[-2] * newDist
                    sumDown += newDist
            
            if (sumDown != 0):
                print(sumUp / sumDown)
            else:
                meanSum = 0.0
                for obj in dataset:
                    meanSum += obj[-2]
                print(meanSum / len(dataset))
                
        else:
            meanSum = 0.0
            cnt = 0
            for obj in dataset:
                if (obj[:-2] == q):
                    cnt += 1
                    meanSum += obj[-2]
            if cnt != 0:
                print(meanSum / cnt)
            else:
                for obj in dataset:
                    meanSum += obj[-2]
                print(meanSum / len(dataset))

    if window == "variable":
        k = int(input())

        sumUp = 0.0
        sumDown = 0.0

        if (dataset[k][-1] != 0):
            for obj in dataset:
                newDist = kernelSmoothing(obj[-1] / dataset[k][-1], kernel)
                sumUp += obj[-2] * newDist
                sumDown += newDist

        if (sumDown != 0):
            print(sumUp / sumDown)
        else:
            meanSum = 0.0
            cnt = 0
            for obj in dataset:
                if (obj[:-2] == q):
                    cnt += 1
                    meanSum += obj[-2]
            if cnt != 0:
                print(meanSum / cnt)
            else:
                for obj in dataset:
                    meanSum += obj[-2]
                print(meanSum / len(dataset))
                

main()
