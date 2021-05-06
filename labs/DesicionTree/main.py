import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

file_to_tree = dict()
path = './DT_csv/'

if __name__ == '__main__':

    acc_min = 1.01
    number_file_min = 0
    tree_depth_min = 1e9
    crit_min = "gini"
    acc_max = 0.0
    number_file_max = 0
    tree_depth_max = 0
    crit_max = "gini"

    best_depths = dict()
    best_accuracy = dict()
    best_crit = dict()
    best_split = dict()

    for tree_depth in tqdm(range(1, 10)):
        for crit in ["gini", "entropy"]:
            for split in ["best", "random"]:
                for file in os.listdir(path):
                    pathToFile = os.path.join(path, file)
                    if ("train" in file):
                        number_file = file[:2]
                        tree = DecisionTreeClassifier(max_depth=tree_depth, criterion=crit, splitter=split)
                        dataset = pd.read_csv(pathToFile)
                        y_train = dataset["y"]
                        x_train = dataset.drop(['y'], axis=1)
                        tree.fit(x_train, y_train)
                        file_to_tree[number_file] = tree

                for file in os.listdir(path):
                    pathToFile = os.path.join(path, file)
                    if ("test" in file):
                        number_file = file[:2]
                        dataset = pd.read_csv(pathToFile)
                        y_test = dataset["y"]
                        x_test = dataset.drop(['y'], axis=1)
                        tree = file_to_tree[number_file]
                        y_pred = tree.predict(x_test)
                        acc = accuracy_score(y_test, y_pred)
                        if (number_file in best_accuracy):
                            if (best_accuracy[number_file] < acc):
                                best_accuracy[number_file] = acc
                                best_depths[number_file] = tree_depth
                                best_crit[number_file] = crit
                                best_split[number_file] = split
                        else:
                            best_accuracy[number_file] = acc
                            best_depths[number_file] = tree_depth
                            best_crit[number_file] = crit
                            best_split[number_file] = split

    for item in best_depths.items():
        number_file, tree_depth = item[0], item[1]
        if (tree_depth > tree_depth_max):
            tree_depth_max = tree_depth
            number_file_max = number_file
        if (tree_depth < tree_depth_min):
            tree_depth_min = tree_depth
            number_file_min = number_file

    print(tree_depth_min, number_file_min, best_crit[number_file_min], best_split[number_file_min])
    print(tree_depth_max, number_file_max, best_crit[number_file_max], best_split[number_file_max])

    list_keys = sorted(list(best_depths.keys()))

    for number_file in list_keys:
        print(f"#{number_file}, best_accuracy: {best_accuracy[number_file]}, tree_depth: {best_depths[number_file]}, criterion: {best_crit[number_file]}, splitter: {best_split[number_file]}")

    def drawGraphic(number_file, tree_depth_sample, type):
        accuracy_train = []
        accuracy_test = []
        depths = []

        crit = best_crit[number_file]
        split = best_split[number_file]

        train_min = path + number_file + "_train.csv"
        test_min = path + number_file + "_test.csv"
        dataset_train_min = pd.read_csv(train_min)
        dataset_test_min = pd.read_csv(test_min)
        y_train = dataset_train_min["y"]
        x_train = dataset_train_min.drop(['y'], axis=1)
        y_test = dataset_test_min["y"]
        x_test = dataset_test_min.drop(['y'], axis=1)

        for tree_depth in range(1, 30):
            tree = DecisionTreeClassifier(max_depth=tree_depth, criterion=crit, splitter=split)
            tree.fit(x_train, y_train)
            y_pred = tree.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            accuracy_test.append(acc)
            y_pred = tree.predict(x_train)
            acc = accuracy_score(y_train, y_pred)
            accuracy_train.append(acc)
            depths.append(tree_depth)

        plt.plot(depths, accuracy_train, color='r', label="train")
        plt.plot(depths, accuracy_test, color='b', label="test")
        plt.xlabel("Depths")
        plt.ylabel("Accuracy")
        plt.title("Graphic for " + type)
        plt.legend()
        plt.show()

    drawGraphic(number_file_min, tree_depth_min, "min height")
    drawGraphic(number_file_max, tree_depth_max, "max height")

    print(list(best_depths.keys()))

    list_keys = sorted(list(best_depths.keys()))

    for number_file in list_keys:

        crit = best_crit[number_file]
        split = best_split[number_file]

        train_min = path + number_file + "_train.csv"
        test_min = path + number_file + "_test.csv"
        dataset_train_min = pd.read_csv(train_min)
        dataset_test_min = pd.read_csv(test_min)
        y_train = dataset_train_min["y"]
        x_train = dataset_train_min.drop(['y'], axis=1)
        y_test = dataset_test_min["y"]
        x_test = dataset_test_min.drop(['y'], axis=1)

        forest = []
        result_train = []
        result_test = []
        y_pred_train = []
        y_pred_test = []

        forest_len = 1000

        for i in range(forest_len):
            tree = DecisionTreeClassifier(max_features="sqrt")
            tree.fit(x_train, y_train)
            forest.append(tree)
            y_pred_train.append(tree.predict(x_train))
            y_pred_test.append(tree.predict(x_test))

        y_pred = []
        for column in range(len(x_train)):
            counts = dict()
            max_count = 0
            max_class = 0
            for row in range(forest_len):
                counts[y_pred_train[row][column]] = counts.get(y_pred_train[row][column], 0) + 1
                if (max_count < counts[y_pred_train[row][column]]):
                    max_count = counts[y_pred_train[row][column]]
                    max_class = y_pred_train[row][column]
            y_pred.append(max_class)

        acc_train = accuracy_score(y_train, y_pred)

        y_pred = []
        for column in range(len(x_test)):
            counts = dict()
            max_count = 0
            max_class = 0
            for row in range(forest_len):
                counts[y_pred_test[row][column]] = counts.get(y_pred_test[row][column], 0) + 1
                if (max_count < counts[y_pred_test[row][column]]):
                    max_count = counts[y_pred_test[row][column]]
                    max_class = y_pred_test[row][column]
            y_pred.append(max_class)

        acc_test = accuracy_score(y_test, y_pred)

        print(f"#{number_file}, acc_train: {acc_train}, acc_test: {acc_test}")