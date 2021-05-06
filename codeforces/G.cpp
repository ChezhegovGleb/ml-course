#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <string>
#include <chrono>

using namespace std;

int M, K, H, N;
int S = 0;
int curFeature = 0;
int countNode = 0;
vector<vector<int> > arr;
vector<string> answer;

struct Node {
    char typeNode{};
    double splitValue{};
    int featureNumber{};
    int clazz{};
    Node* leftNode = nullptr;
    Node* rightNode = nullptr;
};

double indexGini(vector<vector<int> > &a, int indexSplit) {
    int arrSize = a.size();

    vector<double> classes(K);
    double e_left = 0.0;
    for (int i = 0; i < indexSplit; ++i) {
        ++classes[a[i][M]];
    }
    for (int clazz = 0; clazz < K; ++clazz) {
        if (classes[clazz] != 0) {
            e_left += pow(classes[clazz] / indexSplit, 2);
        }
    }

    classes = vector<double>(K, 0);
    double e_right = 0.0;
    for (int i = indexSplit; i < arrSize; ++i) {
        ++classes[a[i][M]];
    }
    for (int clazz = 0; clazz < K; ++clazz) {
        if (classes[clazz] != 0) {
            e_right += pow(classes[clazz] / (arrSize - indexSplit), 2);
        }
    }
    double e_split = (indexSplit * (1 - e_left) + (arrSize - indexSplit) * (1 - e_right)) / arrSize;

    return e_split;
}

double entropy(vector<vector<int> > &a, int indexSplit) {
    int arrSize = a.size();

    vector<double> classes(K);
    double e_left = 0.0;
    for (int i = 0; i < indexSplit; ++i) {
        ++classes[a[i][M]];
    }
    for (int clazz = 0; clazz < K; ++clazz) {
        if (classes[clazz] != 0) {
            e_left += (classes[clazz] / indexSplit) * log2(classes[clazz] / indexSplit);
        }
    }

    classes = vector<double>(K, 0);
    // right child
    double e_right = 0.0;
    for (int i = indexSplit; i < arrSize; ++i) {
        ++classes[a[i][M]];
    }
    for (int clazz = 0; clazz < K; ++clazz) {
        if (classes[clazz] != 0) {
            e_right += (classes[clazz] / (arrSize - indexSplit)) * log2(classes[clazz] / (arrSize - indexSplit));
        }
    }
    double e_split = (indexSplit * -e_left + (arrSize - indexSplit) * -e_right) / arrSize;

    return e_split;
}

bool cmp(vector<int> &first, vector<int> &second) {
    return first[curFeature] < second[curFeature];
}


void buildTree(Node* curNode, int curH, vector<vector<int>> &a) {
    ++S;
    int a_size = a.size();
    vector<int> classes(K, 0);

    unordered_set<int> s;
    for (int i = 0; i < a.size(); ++i) {
        s.insert(a[i][M]);
    }
    if (s.size() == 1) {
        curNode->typeNode = 'C';
        curNode->clazz = *s.begin();
        return;
    }

    if (curH == H) {
        int bestClass = 0;
        int maxCountClass = 0;
        for (int i = 0; i < a.size(); ++i) {
            ++classes[a[i][M]];
            if (classes[a[i][M]] > maxCountClass) {
                maxCountClass = classes[a[i][M]];
                bestClass = a[i][M];
            }
        }
        curNode->typeNode = 'C';
        curNode->clazz = bestClass;
        return;
    }

    double curGain;
    if (a_size < 140) {
        curGain = entropy(a, a.size());
    } else {
        curGain = indexGini(a, a.size());
    }

    double maxGain = 0.0;
    double maxSplit = 0.0;
    int maxIndexSplit = 0;
    int maxFeature = 0;

    vector<vector<int> > bestA = a;

    for (int feature = 0; feature < M; ++feature) {
        curFeature = feature;
        sort(a.begin(), a.end(), cmp);

        vector<int> leftClasses(K);
        vector<int> rightClasses(K);

        double e_left = 0.0;
        double e_right = 0.0;

        for (int i = 0; i < a_size; ++i) {
            ++rightClasses[a[i][M]];
        }
        for (int i = 0; i < K; ++i) {
            e_right += pow(rightClasses[i], 2);
        }

        for (int ind = 1; ind < a_size; ++ind) {
            int indexSplit = ind;
            e_right = e_right - pow(rightClasses[a[ind - 1][M]]--, 2) + pow(rightClasses[a[ind - 1][M]], 2);
            e_left = e_left - pow(leftClasses[a[ind - 1][M]]++, 2) + pow(leftClasses[a[ind - 1][M]], 2);
            double split = (a[ind - 1][feature] + a[ind][feature]) / 2.0;
            double result;
            if (a_size < 140) {
                result = curGain - entropy(a, indexSplit);
            } else {
                result = curGain - (indexSplit * (1 - e_left / pow(indexSplit, 2)) + (a_size - indexSplit) * (1 - e_right / pow(a_size - indexSplit, 2))) / a_size;
            }
            if (result > maxGain) {
                bestA = a;
                maxGain = result;
                maxSplit = split;
                maxFeature = feature;
                maxIndexSplit = indexSplit;
            }
        }
    }

    Node* leftNode = new Node();
    vector<vector<int>> slice(maxIndexSplit);
    copy(bestA.begin(), bestA.begin() + maxIndexSplit, slice.begin());
    buildTree(leftNode, curH + 1, slice);
    Node* rightNode = new Node();
    slice = vector<vector<int>> (bestA.size() - maxIndexSplit);
    copy(bestA.begin() + maxIndexSplit, bestA.end(), slice.begin());
    buildTree(rightNode, curH + 1, slice);

    curNode->typeNode = 'Q';
    curNode->leftNode = leftNode;
    curNode->rightNode = rightNode;
    curNode->featureNumber = maxFeature;
    curNode->splitValue = maxSplit;
}

void printTree(Node* curNode) {
    int curCount = countNode;
    ++countNode;
    string ans;
    if (curNode->typeNode == 'Q') {
        ans = "Q " + to_string(curNode->featureNumber + 1) + ' ' + to_string(curNode->splitValue) + ' ';
        answer.push_back(ans);
        if (curNode->leftNode != nullptr) {
            answer[curCount] += to_string(countNode + 1) + ' ';
            printTree(curNode->leftNode);
        }
        if (curNode->rightNode != nullptr) {
            answer[curCount] += to_string(countNode + 1) + '\n';
            printTree(curNode->rightNode);
        }
    } else {
        ans =  "C " + to_string(curNode->clazz + 1) + '\n';
        answer.push_back(ans);
    }
}


int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> M >> K >> H >> N;

    arr = vector<vector<int> > (N, vector<int>(M + 1));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M + 1; ++j) {
            cin >> arr[i][j];
            if (j == M) {
                --arr[i][j];
            }
        }
    }


    Node* root = new Node();
    buildTree(root, 0, arr);
    cout << S << '\n';
    printTree(root);

    for (const auto& str : answer) {
        cout << str;
    }
}
