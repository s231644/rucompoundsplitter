from collections import deque

data = input().split()
N, M = int(data[0]), int(data[1])

graph = [[] for _ in range(N)]
complexity = [-1 for _ in range(N)]

q = deque()

for i in range(N):
    data_k = list(map(int, input().split()))
    if len(data_k) == 1:
        q.append((i, 0))
    for parent in data_k[1:]:
        graph[parent - 1].append(i)

while q:
    q_top = q.popleft()
    u_top, c_top = q_top
    if complexity[u_top] < c_top:
        complexity[u_top] = c_top
        for ch in graph[u_top]:
            q.append((ch, c_top + 1))

for i in range(N):
    if complexity[i] == M:
        print(i + 1, end=" ")
