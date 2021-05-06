#include <iostream>
#include <vector>
#include <algorithm>

#define int long long

using namespace std;

bool cmp(pair<int, int> &a, pair<int, int> &b) {
    if (a.second == b.second) {
        return a.first < b.first;
    }
    return a.second < b.second;
}

signed main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int k, n;
    cin >> k >> n;

    vector<pair<int, int> > xy(n);
    vector<int> classes(k);

    for (int i = 0; i < n; ++i) {
        cin >> xy[i].first >> xy[i].second;
        --xy[i].second;
        ++classes[xy[i].second];
    }

    sort(xy.begin(), xy.end(), cmp);

    int innerDist = 0;
    int left = 0;
    int right = classes[xy[0].second];

    for (int i = 1; i < n; ++i) {
        if (xy[i].second == xy[i - 1].second) {
            ++left;
            --right;
            int dist = abs(xy[i].first - xy[i - 1].first);
            innerDist += dist * left * right;
        } else {
            left = 0;
            right = classes[xy[i].second];
        }
    }

    innerDist *= 2;
    int allDist = 0;
    left = 0;
    right = n;

    sort(xy.begin(), xy.end());

    for (int i = 1; i < n; ++i) {
        ++left;
        --right;
        int dist = abs(xy[i].first - xy[i - 1].first);
        allDist += dist * left * right;
    }

    allDist *= 2;

    cout << innerDist << '\n';
    cout << allDist - innerDist;
}






 
