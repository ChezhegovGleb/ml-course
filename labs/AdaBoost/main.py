import math

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def drawGraphic(iterate, x, y, forest):
    x_i = 0.0
    step = 0.01

    plt.title(filename + " " + str(iterate + 1))
    plt.xlabel("X")
    plt.ylabel("Y")

    while (x_i < 1.01):
        y_i = 0.0
        while (y_i < 1.01):
            result = 0
            for j in range(iterate + 1):
                result += alphas[j] * forest[j].predict([[x_i, y_i]])[0]
            result = np.sign(result)

            if result <= 0:
                plt.plot(x_i, y_i, color="#ccf5fc", marker="s")
            else:
                plt.plot(x_i, y_i, color="#ffd6d4", marker="s")
            y_i += step
        x_i += step

    for index in range(len(x)):
        if y[index] == 1:
            plt.plot(x[index][0], x[index][1], 'ro')
        else:
            plt.plot(x[index][0], x[index][1], 'bo')

    plt.show()


filename = "chips.csv"

dataset = pd.read_csv(filename)
dataset.loc[dataset['class'] == 'P', 'class'] = 1
dataset.loc[dataset['class'] == 'N', 'class'] = -1

y = list(dataset["class"])
x = dataset.drop(['class'], axis=1)

data_values = dataset.values
minmax = dataset_minmax(data_values)
normalize(data_values, minmax)

xs = []
ys = []

for i in range(len(data_values)):
    xs.append(data_values[i][:-1])
    ys.append(data_values[i][-1])

forest = []
forest_len = 60
m = len(x)

w = [[1 / m for i in range(m)]]

alphas = []
graphic_accuracy = []

numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55]

for i in tqdm(range(forest_len)):
    tree = DecisionTreeClassifier(max_depth=3, criterion="entropy")
    tree.fit(xs, y, sample_weight=w[i])

    y_pred = tree.predict(xs)

    error = 0
    for j in range(m):
        if y_pred[j] != y[j]:
            error += w[i][j]

    alpha = 0.5 * math.log((1 - error) / error)

    tmp_w = [0] * m
    for j in range(m):
        tmp_w[j] = w[i][j] * math.exp(-alpha * y[j] * y_pred[j])
    sum_w = sum(tmp_w)
    for j in range(m):
        tmp_w[j] /= sum_w

    w.append(tmp_w)
    alphas.append(alpha)
    forest.append(tree)

    y_pred = []

    for index, elem in x.iterrows():
        result = 0
        for j in range(i + 1):
            result += alphas[j] * forest[j].predict([elem])[0]
        result = np.sign(result)
        y_pred.append(result)

    graphic_accuracy.append(accuracy_score(y_pred, y))
    if i + 1 in numbers:
        drawGraphic(i, xs, y, forest)

y_pred = []

for index, elem in tqdm(x.iterrows()):
    result = 0
    for i in range(forest_len):
        result += alphas[i] * forest[i].predict([elem])[0]
    result = np.sign(result)
    y_pred.append(result)

print(accuracy_score(y_pred, y))

plt.plot(graphic_accuracy)
plt.xlabel("Iterate")
plt.ylabel("Accuracy")
plt.title("Graphic Accuracy " + filename)
plt.show()