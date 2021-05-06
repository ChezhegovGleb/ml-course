#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int n;
    cin >> n;
    long double sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0;
    for (int i = 0; i < n; ++i) {
        long double x, y;
        cin >> x >> y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        sum_xy += x * y;
    }
    long double numerator = n * sum_xy - sum_x * sum_y;
    long double denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    long double r;
    if (denominator != 0) {
        r = numerator / denominator;
    } else {
        r = 0.0;
    }
    cout << setprecision(9) << r;
}
