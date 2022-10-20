import math
import random

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


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


def create_distance_matrix(points, green, lands):
    """Creates a distance matrix from a list of points."""
    distance_matrix = np.zeros((len(points), len(points)), dtype=np.float32)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):

            for land in lands:
                if get_intersections(points[i], points[j], land):
                    distance_matrix[i, j] = distance_matrix[j, i] = -1
                    break
            else:
                distance = math.dist(points[i], points[j])

                for g in green:
                    for inter in get_intersections(points[i], points[j], g):
                        if len(inter) > 1:
                            distance += math.dist(inter[0], inter[1])

                distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix


class SailingShip(FloatProblem):

    def get_name(self) -> str:
        return "Sailing Ship"

    def __init__(self, ports, sudo_ports, green, lands):
        super().__init__()
        self.ports = ports
        self.sudo_ports = sudo_ports
        self.green = green
        self.lands = lands

        self.number_of_variables = len(sudo_ports) + len(ports)
        self.distance_matrix = create_distance_matrix(ports + sudo_ports, green, lands)

        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = [self.number_of_variables - 1] * self.number_of_variables

        self.number_of_objectives = 2
        self.number_of_constraints = 2
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        fitness = 0
        land_touches = 0
        solution.constraints[0] = 0
        solution.constraints[1] = 1

        ports = set()

        for i in range(self.number_of_variables):
            x = round(solution.variables[i])
            y = round(solution.variables[(i + 1) % self.number_of_variables])

            distance = self.distance_matrix[x][y]
            land_touches += int(x > self.number_of_variables - 1) + int(y > self.number_of_variables - 1)

            if distance == -1:
                fitness = math.inf
                solution.constraints[0] = -1
                break

            if x < len(self.ports):
                ports.add(x)
            if y < len(self.ports):
                ports.add(y)

            fitness += distance

        if len(ports) != len(self.ports):
            solution.constraints[1] = -1

        solution.objectives[0] = fitness
        solution.objectives[1] = land_touches

        return solution

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(lower_bound=self.lower_bound, upper_bound=self.upper_bound,
                                     number_of_objectives=self.number_of_objectives,
                                     number_of_constraints=self.number_of_constraints)
        new_solution.variables = [*range(self.number_of_variables)]
        return new_solution
