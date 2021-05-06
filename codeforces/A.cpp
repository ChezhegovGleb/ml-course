#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
	
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);
	
	int n, m, k;
	cin >> n >> m >> k;
	
	vector<pair<int, int> > v(n);
	
	for (int i = 0; i < n; ++i) {
		int x;
		cin >> x;
		v[i] = {x, i + 1};
	}
	
	sort(v.begin(), v.end());
	
	vector<vector<int> > ans(k);
	int index = 0;
	
	for (int i = 0; i < n; ++i) {
		ans[index].push_back(v[i].second);
		index = (index + 1) % k;
	}
	
	for (int i = 0; i < k; ++i) {
		cout << ans[i].size() << ' ';
		for (int j = 0; j < ans[i].size(); ++j) {
			cout << ans[i][j] << ' ';
		}
		cout << '\n';
	}
	
	return 0;
}
