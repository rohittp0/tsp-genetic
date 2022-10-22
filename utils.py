import math

import numpy as np
from matplotlib import pyplot as plt


def print_solution(my_algo, pop_evolved, logs, params):
    algorithm_name = my_algo.get_name()
    solution_x = pop_evolved.variables
    fitness = pop_evolved.objectives[0]
    n_evals = my_algo.evaluations
    duration = my_algo.total_computing_time
    print('-' * 60)
    print("Function: %s" % "TSP")
    print("Problem dimension: %d" % len(solution_x))
    print("Global Optimum: %d" % 0)
    print('-' * 60)
    print("Algorithm: %s" % algorithm_name)
    print("Parameters:")
    for p in params:
        print("\t%s: " % p, params[p])
    print('-' * 60)
    print("Fitness: %f" % fitness)
    print("Solution: ")
    print(solution_x)
    print('-' * 60)
    print("Nb of functions evaluations: %d" % n_evals)
    print("Stopping criterion: after %d evals" % 200000)
    print("computational time: %.3f seconds" % duration)

    plt.plot(logs[::1000])
    plt.xlabel("evaluations (x1000)")
    plt.ylabel("fitness")
    plt.show()


def get_intersect(a1, a2, b1, b2):
    if min(a1[0], a2[0]) > max(b1[0], b2[0]) or min(b1[0], b2[0]) > max(a1[0], a2[0]):
        return None
    if min(a1[1], a2[1]) > max(b1[1], b2[1]) or min(b1[1], b2[1]) > max(a1[1], a2[1]):
        return None

    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection

    if z == 0:  # lines are parallel
        return None

    return x / z, y / z


def get_intersections(point1, point2, land):
    pts = [(land[0], land[1]), (land[2], land[1]), (land[2], land[3]), (land[0], land[3])]
    lines = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]

    intersections = []

    for line in lines:
        intersect = get_intersect(point1, point2, line[0], line[1])
        if intersect is not None:
            intersections.append(intersect)

    return intersections


def get_closest(origin, point1, point2):
    d1 = math.fabs(origin[0] - point1[0]) + math.fabs(origin[1] - point1[1])
    d2 = math.fabs(origin[0] - point2[0]) + math.fabs(origin[1] - point2[1])

    return point1 if d1 < d2 else point2


def get_path(points, solutions, extended_points):
    path = []
    for i in range(len(solutions) - 1):
        point1 = points[solutions[i]]
        point2 = points[solutions[i + 1]]

        path.append(point1)
        if (point1, point2) in extended_points:
            path.extend(extended_points[(point1, point2)])

    path.append(points[solutions[-1]])

    return path
