#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<vector<int> >  matrix;
vector<int> f;
int m;
int matrixSize;

void buildMatrix() {
	for (int i = 0; i < matrixSize; ++i) {
		int x = i;
		for (int j = 0; j < m; ++j) {
			matrix[i][j] = x % 2;
			x /= 2;
		}
	}
}

int main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);
	
	cin >> m;
	matrixSize = 1 << m;
	matrix = vector<vector<int> >(matrixSize, vector<int>(m));
	
	buildMatrix();
	
	f.resize(matrixSize, 0);
	
	int cnt = 0;
	
	vector<vector<double>> ans;
	
	for (int i = 0; i < matrixSize; ++i) {
		cin >> f[i];
		if (f[i] == 1) {
			ans.push_back(vector<double>(m + 1));
			int cntOne = 0;
			for (int j = 0; j < m; ++j) {
				if (matrix[i][j]) {
					++cntOne;
					ans[cnt][j] = 1; 
				} else {
					ans[cnt][j] = -1;
				}
			}
			ans[cnt][m] = -((double)cntOne - 0.5);
			++cnt;
		}
	}
	
	if (cnt > 512) {
		cnt = 0;
		ans.clear();
	    for (int i = 0; i < matrixSize; ++i) {
			if (f[i] == 0) {
				ans.push_back(vector<double>(m + 1));
				int cntOne = 0;
				for (int j = 0; j < m; ++j) {
					if (matrix[i][j]) {
						++cntOne;
						ans[cnt][j] = 1; 
					} else {
						ans[cnt][j] = -1;
					}
				}
				ans[cnt][m] = -((double)cntOne - 0.5);
				++cnt;
			}
	    }
	    cout << 2 << '\n';
		cout << cnt << ' ' << 1 << '\n';
		
		for (int i = 0; i < cnt; ++i) {
			for (int j = 0; j < m + 1; ++j) {
				cout << ans[i][j] << ' ';
			}
			cout << '\n';
		}
		
		for (int i = 0; i < cnt; ++i) {
			cout << -1 << ' '; 
		}
		cout << 0.5;
	    return 0;
	}
	
	if (cnt == 0) {
		cout << 1 << '\n';
		cout << 1 << '\n';
		for (int i = 0; i < m; ++i) {
			cout << 0 << ' ';
		}
		cout << -0.5;
		return 0;
	}
	
	if (cnt == matrixSize) {
		cout << 1 << '\n';
		cout << 1 << '\n';
		for (int i = 0; i < m; ++i) {
			cout << 0 << ' ';
		}
		cout << 0.5;
		return 0;
	}
	
	cout << 2 << '\n';
	cout << cnt << ' ' << 1 << '\n';
	
	for (int i = 0; i < cnt; ++i) {
		for (int j = 0; j < m + 1; ++j) {
			cout << ans[i][j] << ' ';
		}
		cout << '\n';
	}
	
	for (int i = 0; i < cnt; ++i) {
		cout << 1 << ' '; 
	}
	cout << -0.5;
	
	
	return 0;
}
