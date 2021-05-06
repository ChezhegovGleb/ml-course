#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <unordered_map>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int k;
    cin >> k;

    vector<int> fines(k);
    for (int i = 0; i < k; ++i) {
        cin >> fines[i];
    }

    int alpha;
    cin >> alpha;

    int n;
    cin >> n;

    vector<int> countMessagesInClass(k, 0);
    vector<int> messageToClass(n);
    vector<unordered_map<string, int> > classToTextCountByWord(k);
    unordered_set<string> allWords;

    for (int i = 0; i < n; ++i) {
        unordered_set<string> currentDict;
        int clazz, len;
        cin >> clazz >> len;
        --clazz;

        messageToClass[i] = clazz;
        ++countMessagesInClass[clazz];

        for (int j = 0; j < len; ++j) {
            string word;
            cin >> word;

            allWords.insert(word);
            currentDict.insert(word);
        }

        for (const auto& word : currentDict) {
            if (classToTextCountByWord[clazz].count(word)) {
                classToTextCountByWord[clazz][word] += 1;
            } else {
                classToTextCountByWord[clazz][word] = 1;
            }
        }

    }

    int m;
    cin >> m;

    for (int it = 0; it < m; ++it) {
        vector<long double> results;
        long double sum = 0;
        int len;
        cin >> len;

        unordered_set<string> dictionaryWords;

        for (int j = 0; j < len; ++j) {
            string word;
            cin >> word;
            dictionaryWords.insert(word);
        }

        for (int clazz = 0; clazz < k; ++clazz) {
            long double ans = 1;
            for (const auto& word : allWords) {
                int numerator = alpha;
                int denominator = 2 * alpha;
                if (classToTextCountByWord[clazz].count(word)) {
                    numerator += classToTextCountByWord[clazz][word];
                }

                denominator += countMessagesInClass[clazz];
                long double pwc = (long double) numerator / denominator;

                bool flag = dictionaryWords.count(word);
                ans *= (flag * pwc + (1 - flag) * (1 - pwc));
            }

            long double pc = (long double) countMessagesInClass[clazz] / n;
            ans *= pc * fines[clazz];

            sum += ans;
            results.push_back(ans);
        }

        for (const auto& val : results) {
            cout << fixed << setprecision(9) << fixed << val / sum << ' ';
        }

        cout << '\n';
    }
    return 0;
}
