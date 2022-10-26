import math
import random

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from utils import get_intersections, get_closest


def create_distance_matrix(points, green, lands):
    """Creates a distance matrix from a list of points."""
    distance_matrix = np.zeros((len(points), len(points)), dtype=np.float32)
    extended_points = {}

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.dist(points[i], points[j])

            p1, p2 = points[i], points[j]

            for land in lands:
                inter = get_intersections(p1, p2, land)
                if len(inter) < 2:
                    continue

                distance -= math.dist(inter[0], inter[1])

                (x1, y1), (x2, y2) = get_closest(p1, inter[0], inter[1]), get_closest(p2, inter[0], inter[1])

                a, b, c, d = land[0], land[1], land[2], land[3]
                if x2 == a or y2 == b:
                    a, b, c, d = c, d, a, b

                if (int(x1) == a and int(x2) == c) or (int(y1) == b and int(y2) == d):
                    flip = False
                    if y1 == b and y2 == d:
                        x1, y1, x2, y2 = y1, x1, y2, x2
                        a, b, c, d = b, a, d, c
                        flip = True

                    p = b if math.fabs(b - y1) < math.fabs(d - y1) else d
                    distance += math.fabs(x1 - x2) + math.fabs(2 * p - y1 - y2)

                    extended_points[(i, j)] = [(round(x1), round(y1)), (a, p), (c, p), (round(x2), round(y2))]
                    if flip:
                        extended_points[(i, j)] = [(y, x) for x, y in extended_points[(i, j)]]
                else:
                    distance += math.fabs(x1 - x2) + math.fabs(y1 - y2)
                    extended_points[(i, j)] = [(round(x1), round(y1)), (round(x2), round(y2))]

                extended_points[(j, i)] = [*reversed(extended_points[(i, j)])]

            for g in green:
                inter = get_intersections(points[i], points[j], g)
                if len(inter) > 1:
                    distance += math.dist(inter[0], inter[1])

                distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix, extended_points


class SailingShip(FloatProblem):

    def get_name(self) -> str:
        return "Sailing Ship"

    def __init__(self, ports, green, lands, max_distance):
        super().__init__()
        self.max_distance = max_distance
        self.ports = ports
        self.green = green
        self.lands = lands

        self.number_of_variables = len(ports)
        self.distance_matrix, self.extended_points = create_distance_matrix(ports, green, lands)
        self.fitness = []

        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = [self.number_of_variables - 1] * self.number_of_variables

        self.number_of_objectives = 2
        self.number_of_constraints = self.number_of_variables
        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        fitness = 0
        jumps = 0

        ports = set()
        variables = []

        for i in range(self.number_of_variables):
            p = round(solution.variables[i])

            while p in ports:
                p = (p + 1) % self.number_of_variables
                jumps += 1

            variables.append(p)
            ports.add(p)

        for i in range(self.number_of_variables -1):
            distance = self.distance_matrix[variables[i]][variables[i + 1]]
            fitness += distance
            solution.constraints[i] = self.max_distance - distance

        solution.objectives[0] = fitness
        solution.objectives[1] = jumps
        self.fitness.append(fitness)

        return solution

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(lower_bound=self.lower_bound, upper_bound=self.upper_bound,
                                     number_of_objectives=self.number_of_objectives,
                                     number_of_constraints=self.number_of_constraints)
        new_solution.variables = [*range(self.number_of_variables)]
        return new_solution

    def get_extended_points(self):
        return self.extended_points

    def get_fitness(self):
        return self.fitness
