n = int(input())

xy = [[0, 0]] * n
sum = 0.0

for i in range(n):
    xy[i] = list(map(int, input().split()))

xy.sort()
for i in range(n):
    xy[i][0] = i + 1

xy.sort(key=lambda x: x[1])
sum = 0
for i in range(n):
    xy[i][1] = i + 1
    sum += (xy[i][0] - xy[i][1]) ** 2

p = 1 - 6 * sum / (n * (n ** 2 - 1))

print(p)
