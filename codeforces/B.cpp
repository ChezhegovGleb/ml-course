#include <iostream>
#include <iomanip>

typedef long double ld;

using namespace std;

const int SIZE = 20;

ld cf[SIZE][SIZE];
ld rowSum[SIZE];
ld colSum[SIZE];


int main() {
    int k;
    cin >> k;

    ld all = 0;
    fill(rowSum, rowSum + SIZE, 0);
    fill(colSum, colSum + SIZE, 0);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            cin >> cf[i][j];
            all += cf[i][j];
            rowSum[i] += cf[i][j];
        }
    }

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            colSum[j] += cf[i][j];
        }
    }

    ld microF = 0;
    for (int i = 0; i < k; ++i) {
        ld prec = 0;
        if (rowSum[i] != 0) {
            prec = cf[i][i] / rowSum[i];
        }
        ld rec = 0;
        if (colSum[i] != 0) {
            rec = cf[i][i] / colSum[i];
        }
        if (prec * rec != 0) {
            microF += 2.0 * prec * rec * rowSum[i] / (prec + rec);
        }
    }
    if (all != 0) {
        microF /= all;
    }

    ld precW = 0;
    ld recallW = 0;

    for (int i = 0; i < k; ++i) {
        if (rowSum[i] != 0) {
            precW += rowSum[i] * cf[i][i] / rowSum[i];
        }
    }

    for (int i = 0; i < k; ++i) {
        if (colSum[i] != 0) {
            recallW += rowSum[i] * cf[i][i] / colSum[i];
        }
    }

    ld macroF = 0;
    if (precW + recallW != 0) {
        macroF = 2.0 * precW * recallW / (precW + recallW);
    }
    if (all != 0) {
        macroF /= all;
    }

    cout << setprecision(10);
    cout << macroF << '\n';
    cout << microF << '\n';
}
