#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <numeric>
#include <set>
#include <cstdlib>
#include <time.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr); cout.tie(nullptr);
    int n = 10000000;
    cout << n << '\n';
    long long a;
    srand(time(0));
    for(int i = 0; i < n; i++){
        a = rand();
        cout << a << ' ';
    }
    cout << '\n';
    for(int i = 0; i < n; i++){
        a = rand();
        cout << a << ' ';
    }
    cout << '\n';
}
