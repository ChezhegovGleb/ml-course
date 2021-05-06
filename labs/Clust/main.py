import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

def normalize(array):
    result = array.copy()
    for column in array.columns:
        if column == "class":
            continue
        max_value = array[column].max()
        min_value = array[column].min()
        result[column] = (array[column] - min_value) / (max_value - min_value)

    return result

def pca(x_old):
    covmat = np.cov(x_old.T)
    eigen_values, eigen_vectors = np.linalg.eig(covmat)
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    projection_matrix = (eigen_vectors.T[:][:2]).T
    x_pca = x_old.dot(projection_matrix)
    return x_pca

def getDist(a, b):
    p = 0.0
    for i in range(1, len(a)):
        p += abs(a.iloc[i] - b.iloc[i])
    return p

def zeroing(centroids):
    for i in range(len(centroids)):
        for j in range(len(centroids[i])):
            centroids[i].iloc[j] = 0


def addObjectInClust(i, minClazz, centroids, data):
    centroids[minClazz] += data.iloc[i]

def indexRand(old_data, new_data):
    tp = 0
    fn = 0
    for i in range(len(old_data)):
        for j in range(len(old_data)):
            if i != j:
                if (old_data.loc[i, 'class'] == old_data.loc[j, 'class'] and new_data.loc[i, 'class'] == new_data.loc[j, 'class']):
                    tp += 1
                if (old_data.loc[i, 'class'] != old_data.loc[j, 'class'] and new_data.loc[i, 'class'] != new_data.loc[j, 'class']):
                    fn += 1
    answer = (tp + fn) / (len(old_data) * (len(old_data) - 1))
    return answer

def indexDunn(data):
    denominator = 0
    numerator = float('inf')
    for i in range(len(data)):
        for j in range(len(data)):
            if data.loc[i, 'class'] == data.loc[j, 'class']:
                denominator = max(denominator, getDist(data.iloc[i], data.iloc[j]))
            else:
                numerator = min(numerator, getDist(data.iloc[i], data.iloc[j]))
    answer = numerator / denominator
    return answer


def k_means(data, k):
    centroids = []
    for i in range(k):
        ind = int(random.random() * len(data)) % len(data)
        centroids.append(deepcopy(data.iloc[ind]))


    last_centroids = deepcopy(centroids)
    zeroing(centroids)
    iter = 0

    while True and iter < 5e1:
        iter += 1
        countClazz = [0] * k
        for i in range(len(data)):
            minDist = float('inf')
            minClazz = 0
            order = shuffle(list(range(k)))
            for clazz in order:
                dist = getDist(last_centroids[clazz], dataset.iloc[i]) ** 2
                if dist < minDist:
                    minDist = dist
                    minClazz = clazz
            data.loc[i, 'class'] = minClazz
            countClazz[minClazz] += 1
            addObjectInClust(i, minClazz, centroids, data)

        flag = True
        for i in range(k):
            if countClazz[i] > 0:
                centroids[i] /= countClazz[i]
            for j in range(len(centroids[i])):
                if centroids[i].iloc[j] != last_centroids[i].iloc[j]:
                    flag = False
                    break

        if flag:
            break

        last_centroids = deepcopy(centroids)

    return data, centroids


if __name__ == '__main__':
    filename = "wine.csv"
    dataset = pd.read_csv(filename)
    dataset = normalize(dataset)

    x = np.array(dataset.iloc[:, 1:])
    y = np.array(dataset['class'])

    res = pca(deepcopy(x))
    pca = PCA(n_components=2)
    res = pca.fit_transform(x)

    for i in range(len(dataset)):
        if y[i] == 1:
            plt.plot(res[i][0], res[i][1], "ro")
        if y[i] == 2:
            plt.plot(res[i][0], res[i][1], "go")
        if y[i] == 3:
            plt.plot(res[i][0], res[i][1], "bo")

    plt.show()

    ks = []
    rand = []
    dunn = []

    k = 0
    while k < 16:
        k += 1
        try:
            data, centroids = k_means(deepcopy(dataset), k)
            x = np.array(data.iloc[:, 1:])
            y = np.array(data['class'])
            old_data = deepcopy(dataset)
            new_data = deepcopy(data)
            for i in range(k):
                x = np.concatenate([x, [np.array(centroids[i][1:])]])

            print(x)
            res = pca(deepcopy(x))
            pca = PCA(n_components=2)
            res = pca.fit_transform(x)


            for i in range(len(x) - k):
                if y[i] == 0:
                    plt.plot(res[i][0], res[i][1], "ro")
                if y[i] == 1:
                    plt.plot(res[i][0], res[i][1], "go")
                if y[i] == 2:
                    plt.plot(res[i][0], res[i][1], "bo")

            for i in range(len(x) - k, len(x)):
                plt.plot(res[i][0], res[i][1], "yo")

            plt.show()

            ks.append(k)
            rand.append(indexRand(old_data, new_data))
            dunn.append(indexDunn(new_data))
        except Exception:
            k -= 1
        finally:
            print(k)

    plt.plot(ks, dunn)
    plt.show()
