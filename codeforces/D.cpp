#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <algorithm>

using namespace std;

vector<double> weights;
int n, m;

double h = 1e5;
int iterate = 1e6;
double lambda = 0.9;
double tao = 0.001;
double eps = 1e-9;
double standGrad = 1e-4;


vector<pair<double, double> > dataset_minmax(vector<vector<double> > dataset) {
    vector<pair<double, double> > minmax(m + 2);
    for (int j = 1; j <= m; ++j) {
        double valueMin = dataset[0][j];
        double valueMax = dataset[0][j];
        for (int i = 0; i < n; ++i) {
            valueMin = min(valueMin, dataset[i][j]);
            valueMax = max(valueMax, dataset[i][j]);
        }
        minmax[j] = {valueMin, valueMax};
    }
    return minmax;
}

inline void normalize(vector<vector<double> > &dataset, vector<pair<double, double> > &minmax) {
    for (auto &row : dataset) {
        for (int j = 1; j <= m; ++j) {
            if (minmax[j].second - minmax[j].first == 0) {
                row[j] = 0;
            } else {
                row[j] = (row[j] - minmax[j].first) / (minmax[j].second - minmax[j].first);
            }
        }
    }
}

double sign(double x) {
    if (x > 0)
        return 1.0;
    if (x == 0)
        return 0.0;
    if (x < 0) {
        return -1.0;
    }
}

inline void sgd(vector<vector<double> > &dataset) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i <= m; ++i) {
        weights[i] = 0;
//        weights[i] = ((double)rand() / RAND_MAX * 2 - 1) / (2 * m);
    }

    vector<double> moment(m + 1);

    for (int d = 3; d <= iterate; ++d) {
        int index = rand() % n;
        double curY = 0;
        double realY = dataset[index][m + 1];

        vector<double> grad(m + 1, 0);

        for (int j = 0; j <= m; ++j) {
            curY += weights[j] * dataset[index][j];
        }

        double diffY = curY - realY;

        double stepReducer = log2(d);
        double curH = h / stepReducer;
        double gradSmape = sign(diffY) / (abs(curY) + abs(realY) + eps) - (sign(curY) * abs(diffY) / (pow(abs(curY) + abs(realY), 2) + eps));

        for (int i = 0; i <= m; ++i) {
            grad[i] = dataset[index][i] * gradSmape;
            if (abs(grad[i]) > 1e6) {
                grad[i] = sign(grad[i]);
            }
            weights[i] -= curH * grad[i];
        }
    }
}

inline void denormalize(vector<double> &oldWeights, vector<pair<double, double> > &minmax) {
    for (int i = 1; i <= m; ++i) {
        if (minmax[i].second - minmax[i].first != 0) {
            oldWeights[0] -= oldWeights[i] / (minmax[i].second - minmax[i].first) * minmax[i].first;
            oldWeights[i] /= (minmax[i].second - minmax[i].first);
        } else {
            oldWeights[i] = 0;
        }
    }
}


int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    cin >> n >> m;

    vector<vector<double> > dataset(n, vector<double>(m + 2));
    weights.resize(m + 1);
    double sumY = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= m + 1; ++j) {
            if (j == 0) {
                dataset[i][j] = 1;
            } else {
                int x;
                cin >> x;
                dataset[i][j] = x;
                if (j == m + 1) {
                    sumY += abs(x);
                }
            }
        }
    }

    if (m == 1) {
        h = min(h, sumY / n);
    }

    auto minmax = dataset_minmax(dataset);

    normalize(dataset, minmax);

    sgd(dataset);

    denormalize(weights, minmax);

    for (int i = 1; i <= m; ++i) {
        cout << weights[i] << '\n';
    }
    cout << weights[0] << '\n';
}
