import math

k = int(input())
n = int(input())

xs = [0] * n
ys = [0] * n

countX = [0] * k

realObserv = dict()

for i in range(n):
    xs[i], ys[i] = map(int, input().split())
    xs[i] -= 1
    ys[i] -= 1
    countX[xs[i]] += 1
    realObserv[(xs[i], ys[i])] = realObserv.get((xs[i], ys[i]), 0) + 1

e = [0] * k
e2 = [0] * k

for x, y in realObserv:
    p_y_x = realObserv[(x, y)] / countX[x]
    e[x] += y ** 2 * p_y_x
    e2[x] += y * p_y_x

ans = 0.0
for x in range(k):
    p_x = countX[x] / n
    ans += p_x * (e[x] - e2[x] ** 2)

print(ans)
