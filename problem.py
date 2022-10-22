import math
import random

import numpy as np
from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution

from utils import get_intersections, get_closest


def create_distance_matrix(points, green, lands):
    """Creates a distance matrix from a list of points."""
    distance_matrix = np.zeros((len(points), len(points)), dtype=np.float32)
    extended_points = {}

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.dist(points[i], points[j])

            for land in lands:
                inter = get_intersections(points[i], points[j], land)
                if len(inter) < 2:
                    continue

                distance -= math.dist(inter[0], inter[1])

                x1, y1, x2, y2 = inter[0][0], inter[0][1], inter[1][0], inter[1][1]

                x1, y1 = get_closest(points[i], (x1, y1), (x2, y2))
                x2, y2 = get_closest(points[j], (x1, y1), (x2, y2))

                a, b, c, d = land[0], land[1], land[2], land[3]

                a, b = get_closest(points[i], (a, b), (c, d))
                c, d = get_closest(points[j], (a, b), (c, d))

                path = math.fabs(x1 - x2)

                if (x1 == a and x2 == c) or (y1 == b and y2 == d):
                    if y1 == b and y2 == d:
                        x1, y1, x2, y2 = y1, x1, y2, x2
                        a, b, c, d = b, a, d, c

                    p = b if math.fabs(b - y1) < math.fabs(d - y1) else d
                    path += math.fabs(2 * p - y1 - y2)

                    extended_points[(i, j)] = [(x1, y1), (a, p), (c, p), (x2, y2)]
                    extended_points[(j, i)] = [(x2, y2), (c, p), (a, p), (x1, y1)]
                else:
                    path += math.fabs(y1 - y2)

                    extended_points[(i, j)] = [(x1, y1), (x2, y2)]
                    extended_points[(j, i)] = [(x2, y2), (x1, y1)]

            for g in green:
                inter = get_intersections(points[i], points[j], g)
                if len(inter) > 1:
                    distance += math.dist(inter[0], inter[1])

                distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix, extended_points


class SailingShip(PermutationProblem):

    def get_name(self) -> str:
        return "Sailing Ship"

    def __init__(self, ports, green, lands):
        super().__init__()
        self.ports = ports
        self.green = green
        self.lands = lands

        self.number_of_variables = len(ports)
        self.distance_matrix, self.extended_points = create_distance_matrix(ports, green, lands)
        self.fitness = []

        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = [self.number_of_variables - 1] * self.number_of_variables

        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE]

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        fitnes = 0

        for i in range(self.number_of_variables - 1):
            x = solution.variables[i]
            y = solution.variables[(i + 1)]

            distance = self.distance_matrix[x][y]

            fitnes += distance

        solution.objectives[0] = fitnes
        self.fitness.append(fitnes)

        return solution

    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives,
                                           number_of_constraints=self.number_of_constraints)
        new_solution.variables = random.sample(range(self.number_of_variables), k=self.number_of_variables)
        return new_solution

    def get_extended_points(self):
        return self.extended_points

    def get_fitness(self):
        return self.fitness
