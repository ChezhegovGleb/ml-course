import math

kx, ky = map(int, input().split())
n = int(input())

xs = [0] * n
ys = [0] * n

countX = [0] * kx
countY = [0] * ky

realObserv = dict()

for i in range(n):
    xs[i], ys[i] = map(int, input().split())
    xs[i] -= 1
    ys[i] -= 1
    countX[xs[i]] += 1
    countY[ys[i]] += 1
    realObserv[(xs[i], ys[i])] = realObserv.get((xs[i], ys[i]), 0) + 1

ans = 0.0
for x, y in realObserv:
    p_y_x = realObserv[(x, y)] / countX[x]
    p_x = countX[x] / n
    ans -= p_x * p_y_x * math.log(p_y_x)

print(ans)
