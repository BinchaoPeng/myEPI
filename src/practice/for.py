a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
n = 5
for b in [a[i:i + n] for i in range(0, len(a), n)]:
    print(b)

for i in range(0, len(a), n):
    print(a[i:i + n])
