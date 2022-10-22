import cv2
import numpy as np
from jmetal.algorithm.multiobjective import MOEAD
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

from problem import SailingShip
from utils import print_solution, get_path


def get_variables():
    ports = [(90, 17), (157, 102), (38, 123), (80, 143), (96, 44), (23, 80), (8, 40), (66, 73), (86, 81), (181, 165),
             (9, 173), (287, 241)]
    lands = [(20, 20, 40, 45), (100, 50, 150, 70)]
    green = [(50, 100, 80, 120), (200, 56, 250, 80)]

    map_dimensions = 300

    sea_map = np.ones((map_dimensions, map_dimensions, 3), dtype=np.uint8) * 255

    return sea_map, ports, lands, green


def draw_map(sea_map, ports, lands, green):
    """Draws a map of the world."""

    for port in ports:
        cv2.drawMarker(sea_map, port, (255, 0, 0), markerType=cv2.MARKER_DIAMOND, markerSize=4, thickness=2)

    for land in lands:
        cv2.rectangle(sea_map, (land[0], land[1]), (land[2], land[3]), (0, 0, 255), -1, lineType=cv2.LINE_AA)

    for g in green:
        cv2.rectangle(sea_map, (g[0], g[1]), (g[2], g[3]), (0, 255, 0), -1, lineType=cv2.LINE_AA)

    return sea_map


def run(problem):
    max_eval = 500_000

    pop_size = 600
    offspring = 600
    mut_prob = 0.04
    cross_prob = 0.5

    algorithm = MOEAD(
        problem=problem,
        population_size=3000,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path='resources/MOEAD_weights',
        termination_criterion=StoppingByEvaluations(max_eval)
    )

    progress_bar = ProgressBarObserver(max=max_eval)
    algorithm.observable.register(progress_bar)

    algorithm.run()
    result = algorithm.get_result()[0]

    params = {'population': pop_size,
              'offspring': offspring,
              'mutation probability': mut_prob,
              'crossover probability': cross_prob,
              }

    variables = []
    ports = set()

    for i in range(len(result.variables)):
        p = round(result.variables[i])

        while p in ports:
            p = (p + 1) % len(result.variables)

        variables.append(p)
        ports.add(p)

    print_solution(algorithm, result, problem.get_fitness(), params)

    return variables


def draw_solution(sea_map, solution, points, extended_points):
    coords = get_path(points, solution, extended_points)
    for i in range(len(coords) - 1):
        cv2.line(sea_map, coords[i], coords[i + 1], (0, 0, 0), lineType=cv2.LINE_AA)

    return sea_map


def main():
    sea_map, ports, lands, green = get_variables()

    problem = SailingShip(ports, green, lands)

    solution = run(problem)
    extended_points = problem.extended_points

    if solution is None:
        print("No solution found")
        return

    sea_map = draw_map(sea_map, ports, lands, green)
    sea_map = draw_solution(sea_map, solution, ports, extended_points)

    cv2.imshow('map', sea_map)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
