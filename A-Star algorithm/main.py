# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import math
import heapq

matrix = [[0 for x in range(100)] for y in range(100)]
N = 100

f = open("surface_100x100.txt", "r")
fl = f.readlines()
for point in fl:
    (x, y, z, b) = [t(s) for t, s in zip((int, int, float, int), point.split())]
    if b == 0:
        matrix[x][y] = z
    else:
        matrix[x][y] = -z

x_source = []
y_source = []
x_destination = []
y_destination = []

line_index = 0
f = open("surface_100x100.end_points.txt", "r")
fl = f.readlines()
for point in fl:
    if line_index == 0:
        x_source.append(int(point.split()[0]))
        y_source.append(int(point.split()[1]))
        line_index = 1
    else:
        x_destination.append(int(point.split()[0]))
        y_destination.append(int(point.split()[1]))
        line_index = 0


# Validation of a node, if not an obstacle and valid position
def is_valid(x, y):
    if 0 <= x < N and 0 <= y < N and matrix[x][y] >= 0.0:
        return True
    else:
        return False


# Heuristics
def euclidean_distance_3d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (matrix[x2][y2] - matrix[x1][y1]) ** 2)


def chebyshev_distance(x1, x2, y1, y2):
    return max(abs(y2 - y1), abs(x2 - x1))


# shortest path using A* algorithm
def a_star_3d(source, destination):
    # aux vectors for generating neighbours
    neigh_aux = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_list = []
    closed_list = []
    out_path = []
    # nodes of the path
    path_ = {}

    # f(x) = g(x) + h(x), where:
    # g(x) = length of the path from S source to x node
    # h(x) = heuristic, estimated length of the path from x to G destination

    g = {source: 0}
    f_cost = {source: euclidean_distance_3d(source[0], source[1], destination[0], destination[1])}

    if not is_valid(source[0], source[1]):
        return -1

    if not is_valid(destination[0], destination[1]):
        return -1

    # heap for retrieving the node with the smallest cost
    heapq.heappush(open_list, (f_cost[source], source))

    while open_list:

        q = heapq.heappop(open_list)[1]

        if q == destination:
            while q in path_:
                out_path.append(q)
                q = path_[q]
            return g[destination], out_path

        closed_list.append(q)
        for i in neigh_aux:
            neighbour = (q[0] + i[0], q[1] + i[1])
            if is_valid(neighbour[0], neighbour[1]):

                curr_g = g[q] + euclidean_distance_3d(q[0], q[1], neighbour[0], neighbour[1])

                # worse path
                if neighbour in closed_list:
                    continue

                # found a better path or not yet included in the best path
                if curr_g < g.get(neighbour, 0) or neighbour not in [i[1] for i in open_list]:
                    path_[neighbour] = q
                    g[neighbour] = curr_g
                    f_cost[neighbour] = curr_g + euclidean_distance_3d(neighbour[0], neighbour[1],
                                                                       destination[0], destination[1])
                    heapq.heappush(open_list, (f_cost[neighbour], neighbour))
            else:
                continue


# path from S to G with minimum number of nodes between
def a_star_3d_min_nodes(source, destination):
    neigh_aux = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_list = []
    closed_list = []
    out_path = []
    path_ = {}

    # f(x) = g(x) + h(x), where:
    # g(x) = length of the path from S source to x node
    # h(x) = heuristic, estimated length of the path from x to G destination

    g = {source: 0}
    f_cost = {source: chebyshev_distance(source[0], source[1], destination[0], destination[1])}

    if not is_valid(source[0], source[1]):
        return -1

    heapq.heappush(open_list, (f_cost[source], source))

    while open_list:

        q = heapq.heappop(open_list)[1]

        if q == destination:
            while q in path_:
                out_path.append(q)
                q = path_[q]
            return g[destination], out_path

        closed_list.append(q)
        for i in neigh_aux:
            neighbour = (q[0] + i[0], q[1] + i[1])
            if is_valid(neighbour[0], neighbour[1]):

                curr_g = g[q] + 1

                if neighbour in closed_list:
                    continue

                if curr_g < g.get(neighbour, 0) or neighbour not in [i[1] for i in open_list]:
                    path_[neighbour] = q
                    g[neighbour] = curr_g
                    f_cost[neighbour] = curr_g + chebyshev_distance(neighbour[0], neighbour[1], destination[0], destination[1])
                    heapq.heappush(open_list, (f_cost[neighbour], neighbour))
            else:
                continue


if __name__ == '__main__':
    (path_length, path) = a_star_3d((x_source[0], y_source[0]), (x_destination[0], y_destination[0]))
    print("\nShortest path length: " + str(path_length))
    print("Path: " + str(path) + "\n")
    (path_length, path) = a_star_3d_min_nodes((x_source[0], y_source[0]), (x_destination[0], y_destination[0]))
    print("\nNumber of nodes in path with minimum number of nodes: " + str(path_length))
    print("Path: " + str(path) + "\n")

