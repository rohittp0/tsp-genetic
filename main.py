import cv2
import numpy as np
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import SimpleRandomMutation
from jmetal.util import termination_criterion
from jmetal.util.observer import ProgressBarObserver

from problem import SailingShip
from utils import print_solution


def variables():
    ports = [(10, 10), (290, 290), (100, 30), (10, 200)]
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


def get_sudo_ports(lands):
    ports = []

    for land in lands:
        ports.append((land[0], land[1]))
        ports.append((land[2], land[3]))
        ports.append((land[0], land[3]))

    return ports


def run(problem):
    max_eval = 300_000

    pop_size = 40_000
    offspring = 4000
    mut_prob = 0.05
    cross_prob = 0.9

    termination = termination_criterion.StoppingByEvaluations(max_evaluations=max_eval)

    algorithm = NSGAII(
        problem=problem,
        population_size=pop_size,
        offspring_population_size=offspring,
        mutation=SimpleRandomMutation(mut_prob),
        crossover=SBXCrossover(cross_prob),
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
    result = [r for r in algorithm.get_result() if sum(r.constraints) >= 0]

    if len(result) == 0:
        return None

    params = {'population': pop_size,
              'offspring': offspring,
              'mutation probability': mut_prob,
              'crossover probability': cross_prob,
              }

    print_solution(algorithm, result, params)

    result = [round(v) for v in result[0].variables]
    result = [*dict.fromkeys(result)]

    return result


def draw_solution(sea_map, solution, points):
    coords = [points[round(i)] for i in solution]
    for i in range(len(coords)):
        cv2.line(sea_map, coords[i], coords[(i+1) % len(coords)], (0, 0, 0), lineType=cv2.LINE_AA)

    return sea_map


def main():
    sea_map, ports, lands, green = variables()
    sudo_ports = get_sudo_ports(lands)

    problem = SailingShip(ports, sudo_ports, green, lands)

    solution = run(problem)

    if solution is None:
        print("No solution found")
        return

    sea_map = draw_map(sea_map, ports, lands, green)
    sea_map = draw_solution(sea_map, solution, ports + sudo_ports)

    cv2.imshow('map', sea_map)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
