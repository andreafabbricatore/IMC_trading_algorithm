import itertools


def calc_profit(path):
    profit = 1
    for i in range(5):
        profit *= prices[path[i]][path[i + 1]]
    return profit


prices = [[1, 0.5, 1.45, 0.75], [1.95, 1, 3.1, 1.49], [0.67, 0.31, 1, 0.48], [1.34, 0.64, 1.98, 1]]

paths = itertools.product([0, 1, 2, 3], repeat=4)
full_paths = list()
for path in paths:
    full_paths.append([3] + list(path) + [3])

max_profit = 0
best_path = None
for path in full_paths:
    profit = calc_profit(path)
    if profit > max_profit:
        max_profit = profit
        best_path = path

print(best_path, max_profit)



