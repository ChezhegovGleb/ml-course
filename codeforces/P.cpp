#include <iostream>
#include <unordered_map>
#include <set>
#include <vector>
#include <iomanip>

#define int long long

using namespace std;

signed main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int k1, k2, n;
    cin >> k1 >> k2 >> n;

    unordered_map<int, unordered_map<int, double> > realObserv;

    vector<int> x_sum(k1);
    vector<int> y_sum(k2);
    set<pair<int, int> > xy;

    for (int i = 0; i < n; ++i) {
        int x, y;
        cin >> x >> y;
        --x;
        --y;
        xy.insert({x, y});
        ++x_sum[x];
        ++y_sum[y];
        realObserv[x][y] = realObserv[x][y] + 1.0;
    }

    long double hi2 = 0.0;

    vector<long double> countX(k1);

    for (auto p : xy) {
        int i = p.first;
        int j = p.second;
        long double expectObserv = x_sum[i] * y_sum[j] / (long double)n;
        hi2 += (realObserv[i][j] - expectObserv) * (realObserv[i][j] - expectObserv) / expectObserv;
        countX[i] += expectObserv;
    }

    for (int i = 0; i < k1; ++i) {
        hi2 += x_sum[i] - countX[i];
    }

    cout << fixed << setprecision(9) << hi2;
}
