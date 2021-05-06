#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int n;
double C;
double b = 0.0;
double eps = 1e-3;

vector<double> alphas;
vector<vector<int> > dataset;
vector<double> target;

int takeStep(int i1, int i2) {
    if (i1 == i2) {
        return 0;
    }

    double alph1 = alphas[i1];
    double alph2 = alphas[i2];
    double y1 = target[i1];
    double y2 = target[i2];
    double E1 = -b - y1;
    double E2 = -b - y2;
    for (int i = 0; i < n; ++i) {
        E1 += alphas[i] * dataset[i1][i] * target[i];
        E2 += alphas[i] * dataset[i2][i] * target[i];
    }
    double s = y1 * y2;
    double L, H;
    if (y1 == y2) {
        L = max(0.0, alph2 + alph1 - C);
        H = min(C, alph2 + alph1);
    } else {
        L = max(0.0, alph2 - alph1);
        H = min(C, C + alph2 - alph1);
    }

    if (L == H) {
        return 0;
    }

    double k11 = dataset[i1][i1];
    double k12 = dataset[i1][i2];
    double k21 = dataset[i2][i1];
    double k22 = dataset[i2][i2];
    double eta = k11 + k22 - 2 * k12;

    double a2;
    if (eta > 0) {
        a2 = alph2 + y2 * (E1 - E2) / eta;
        if (a2 < L) {
            a2 = L;
        } else if (a2 > H) {
            a2 = H;
        }
    } else {
        a2 = H;
    }

    if (abs(a2 - alph2) < eps * (a2 + alph2 + eps)) {
        return 0;
    }

    double a1 = alph1 + s * (alph2 - a2);

    b = ((E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + b) +
            (E2 + y1 * (a1 - alph1) * k21 + y2 * (a2 - alph2) * k22 + b)) / 2;

    alphas[i1] = a1;
    alphas[i2] = a2;
    return 1;
}

int examineExample(int i2) {
    double y2 = target[i2];
    double alph2 = alphas[i2];

    double E2 = -b - y2;
    for (int i = 0; i < n; ++i) {
        E2 += alphas[i] * dataset[i2][i] * target[i];
    }
    double r2 = E2 * y2;

    if ((r2 < -eps && alph2 < C) || (r2 > eps && alph2 > 0)) {
        for (int i = 0; i < n * 3; ++i) {
            if (alphas[i] != 0 && alphas[i] != C) {
                int i1 = rand() % n;
                if (takeStep(i1, i2)) {
                    return 1;
                }
            }
        }
    }

    return 0;
}


int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    cin >> n;

    alphas = vector<double>(n, 0.0);
    dataset = vector<vector<int> > (n, vector<int>(n + 1));
    target = vector<double>(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            cin >> dataset[i][j];
            if (j == n) {
                target[i] = dataset[i][j];
            }
        }
    }

    cin >> C;

    int numChanged = 0;
    int examineAll = 1;

    while (numChanged > 0 || examineAll) {
        numChanged = 0;
        if (examineAll) {
            for (int i = 0; i < n; ++i) {
                numChanged += examineExample(i);
            }
        } else {
            for (int i = 0; i < n; ++i) {
                if (alphas[i] != 0 && alphas[i] != C) {
                    numChanged += examineExample(i);
                }
            }
        }
        if (examineAll == 1) {
            examineAll = 0;
        } else if (numChanged == 0) {
            examineAll = 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        cout << alphas[i] << ' ';
    }
    cout << -b << '\n';
}
