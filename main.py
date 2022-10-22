import cv2
import numpy as np
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.util import termination_criterion
from jmetal.util.observer import ProgressBarObserver

from problem import SailingShip
from utils import print_solution, get_path


def variables():
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
    max_eval = 15_000.3

    pop_size = 400
    offspring = 400
    mut_prob = 0.05
    cross_prob = 0.9

    termination = termination_criterion.StoppingByEvaluations(max_evaluations=max_eval)

    algorithm = NSGAII(
        problem=problem,
        population_size=pop_size,
        offspring_population_size=offspring,
        mutation=PermutationSwapMutation(mut_prob),
        crossover=PMXCrossover(cross_prob),
        termination_criterion=termination,
    )

    # algorithm = GeneticAlgorithm(
    #     problem=problem,
    #     population_size=pop_size,
    #     offspring_population_size=offspring,
    #     mutation=PermutationSwapMutation(mut_prob),
    #     crossover=PMXCrossover(cross_prob),
    #     selection=select,
    #     termination_criterion=termination
    # )

    progress_bar = ProgressBarObserver(max=max_eval)
    algorithm.observable.register(progress_bar)

    algorithm.run()
    result = algorithm.get_result()[0]

    params = {'population': pop_size,
              'offspring': offspring,
              'mutation probability': mut_prob,
              'crossover probability': cross_prob,
              }

    print_solution(algorithm, result, problem.get_fitness(), params)

    return result.variables


def draw_solution(sea_map, solution, points, extended_points):
    coords = get_path(points, solution, extended_points)
    for i in range(len(coords) - 1):
        cv2.line(sea_map, coords[i], coords[i + 1], (0, 0, 0), lineType=cv2.LINE_AA)

    return sea_map


def main():
    sea_map, ports, lands, green = variables()

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
