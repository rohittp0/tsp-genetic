import math
import time
from functools import partial

import numpy as np
from jmetal.algorithm.multiobjective import NSGAII, MOEAD
from jmetal.config import store
from jmetal.core.algorithm import Algorithm

from typing import TypeVar

from jmetal.lab.experiment import Job
from jmetal.operator import PolynomialMutation, SBXCrossover, DifferentialEvolutionCrossover
from jmetal.problem import ZDT1
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.evaluator import MapEvaluator
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from sklearn.cluster import KMeans

S = TypeVar('S')
R = TypeVar('R')


def Clustering_kmeans(population, number_of_collectives):
    kmeans = KMeans(n_clusters=number_of_collectives, max_iter=1000)

    pop_temp = []
    for solution in population:
        pop_temp.append(solution.variables)

    return kmeans.fit_predict(pop_temp) + 1


def Clustering(population, number_of_collectives, problem):
    return generate_labels(population, number_of_collectives)


def generate_labels(population, number_of_collectives):
    individual_average = [sum(s.variables) / len(population) for s in population]
    avg = sum(individual_average) / len(individual_average)
    sd = np.std(individual_average)

    # todo refactor
    labels = [0 for i in range(len(population))]
    if number_of_collectives == 4:
        for i in range(len(population)):
            if individual_average[i] < avg - 0.6 * sd:
                labels[i] = 1
            elif individual_average[i] < avg:
                labels[i] = 2
            elif individual_average[i] < avg + 0.6 * sd:
                labels[i] = 3
            else:
                labels[i] = 4

    elif number_of_collectives == 6:
        for i in range(len(population)):
            if individual_average[i] < avg - sd:
                labels[i] = 1
            elif individual_average[i] < avg - sd / 2:
                labels[i] = 2
            elif individual_average[i] < avg:
                labels[i] = 3
            elif individual_average[i] < avg + sd / 2:
                labels[i] = 4
            elif individual_average[i] < avg + sd:
                labels[i] = 5
            else:
                labels[i] = 6

    elif number_of_collectives == 8:
        for i in range(len(population)):
            if individual_average[i] < avg - 1.1 * sd:
                labels[i] = 1
            elif individual_average[i] < avg - 0.6 * sd:
                labels[i] = 2
            elif individual_average[i] < avg - 0.2 * sd:
                labels[i] = 3
            elif individual_average[i] < avg:
                labels[i] = 4
            elif individual_average[i] < avg + 0.2 * sd:
                labels[i] = 5
            elif individual_average[i] < avg + 0.6 * sd:
                labels[i] = 6
            elif individual_average[i] < avg + 1.1 * sd:
                labels[i] = 7
            else:
                labels[i] = 8

    else:
        print("Number of collectives: {} not supported.".format(number_of_collectives))
        exit()

    return labels


class Collective(object):

    def __init__(self, algorithm, label):
        self.algorithm = algorithm
        self.label = label

        self.evaluations = 0
        self.solutions = []

    def add_solution(self, solution, max_number=0):
        if max_number != 0:
            if len(self.solutions) < max_number:
                self.solutions.append(solution)
        else:
            self.solutions.append(solution)

    def erase(self):
        self.solutions = []
        self.algorithm.solutions = []

    def restart(self):
        self.algorithm.solutions = self.solutions
        # self.algorithm.init_progress()
        # self.algorithm.evaluations += self.evaluations

    def step(self):
        self.algorithm.step()
        self.evaluations = self.algorithm.evaluations

    def evaluate(self):
        return self.algorithm.evaluate(self.algorithm.solutions)

    def calculate_fitness(self):
        self.solutions = self.algorithm.solutions

        # Average fitness of all solutions (MLS1)
        collective_fitness = 0
        for solution in self.solutions:
            temp_fitness = 0
            for objective in solution.objectives:
                temp_fitness += objective / len(solution.objectives)
            collective_fitness += temp_fitness
        collective_fitness /= len(self.solutions)

        return collective_fitness

    # TODO sort and get by fitness
    def best_solutions(self, num_solutions):
        return self.algorithm.solutions[:num_solutions]

    def init_algorithm(self, *args):
        constructor, kwargs = self.algorithm(*args)

        self.algorithm = constructor(**kwargs)
        self.algorithm.solutions = self.solutions

    def __repr__(self):
        return "Collective {} - {} - {} solutions".format(
            self.label, self.algorithm, len(self.algorithm.solutions))


class MultiLevelSelection(Algorithm[S, R]):

    def __init__(self,
                 problem=ZDT1(),
                 number_of_collectives=6,
                 num_new_collectives=2,
                 population_size=600,
                 algorithms=None,
                 mls_mode=7,
                 max_evaluations=30000,
                 termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator,
                 population_evaluator=store.default_evaluator):
        super(MultiLevelSelection, self).__init__()
        if algorithms is None:
            algorithms = []
        self.problem = problem
        self.number_of_collectives = number_of_collectives
        self.num_new_collectives = num_new_collectives
        self.population_size = population_size

        self.algorithms = algorithms
        self.coevolution_amount = len(algorithms)
        self.mls_mode = mls_mode

        self.max_evaluations = max_evaluations
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.generations = 0

        self.solutions = []
        self.collectives = self.generate_collectives()
        self.active_collective = None

    def generate_collectives(self):
        collectives = self.initialise_temp_collectives()

        # mating pool size must be even for jmetalpy
        for col in collectives:
            if len(col.solutions) % 2 != 0 and hasattr(col.algorithm, "mating_pool_size"):
                col.algorithm.mating_pool_size += 1

        return collectives

    def initialise_temp_collectives(self):
        subpopulation_sizes = self.get_subpopulation_sizes()

        collectives = []
        for i in range(0, self.coevolution_amount):
            algorithm = self.algorithms[i]
            collective_size = subpopulation_sizes[i]

            population = [
                self.population_generator.new(self.problem)
                for _ in range(collective_size)
            ]

            for collective in self.assign_collectives(algorithm, population):
                collectives.append(collective)

        return collectives

    def assign_collectives(self, algorithm, population):
        labels = self.cluster_population(population)

        collectives = []
        for unique_label in set(labels):

            collective = Collective(algorithm, unique_label)
            for label, solution in zip(labels, population):
                if label == unique_label:
                    collective.add_solution(solution)

            collective.init_algorithm(self.problem,
                                      len(collective.solutions),
                                      self.max_evaluations,
                                      self.population_evaluator)

            collectives.append(collective)

        return collectives

    def cluster_population(self, population):
        if self.number_of_collectives > 1:
            labels = Clustering(population, self.number_of_collectives, self.problem)
        elif self.number_of_collectives == 1:
            labels = [1 for _ in range(0, self.population_size)]
        else:
            raise SystemError("Error: number of collectives < 1")

        return labels

    def get_subpopulation_sizes(self):
        subpopulation_sizes = [
            int(self.population_size / self.coevolution_amount)
            for _ in range(0, self.coevolution_amount)
        ]

        # Spread remainder over the subpopulations
        for i in range(0, self.population_size % self.coevolution_amount):
            subpopulation_sizes[i] += 1

        return subpopulation_sizes

    def create_initial_solutions(self):
        self.collectives = self.generate_collectives()

        solutions = []
        for collective in self.collectives:
            solutions.extend(collective.algorithm.solutions)

        return solutions

    def evaluate(self, solution_list):
        # TODO refactor, convert to list if given a single solution
        if not isinstance(solution_list, list):
            solution_list = [solution_list]

        solutions = []
        if self.active_collective:
            solutions = self.active_collective.evaluate()
            self.active_collective = None
        else:
            for collective in self.collectives:
                solutions.extend(collective.evaluate())

        return solutions

    def init_progress(self):
        for collective in self.collectives:
            collective.algorithm.init_progress()

    def step(self):
        self._update_collectives()
        for i in range(self.num_new_collectives):
            self._replace_worst_collective()

    def _update_collectives(self):
        solutions = []
        for collective in self.collectives:
            collective.step()

            self.active_collective = collective
            solutions.extend(collective.algorithm.solutions)

        self.solutions = solutions

    def _replace_worst_collective(self):
        worst_collective = self._get_worst_collective()
        worst_collective_size = len(worst_collective.solutions)

        worst_collective.erase()
        self.collectives.remove(worst_collective)

        num_solutions = math.ceil(worst_collective_size / len(self.collectives))
        for collective in self.collectives:
            for best_solution in collective.best_solutions(num_solutions):
                worst_collective.add_solution(best_solution, worst_collective_size)

        worst_collective.restart()
        self.collectives.append(worst_collective)

    def _get_worst_collective(self):
        worst_collective = None
        fitness = 1e10

        for collective in self.collectives:
            collective_fitness = collective.calculate_fitness()
            if collective_fitness < fitness:
                worst_collective = collective
                fitness = collective_fitness

        # print("Replace {}, fitness: {}".format(worst_collective, fitness))
        return worst_collective

    def update_progress(self):
        evaluations = 0
        for collective in self.collectives:
            collective.algorithm.update_progress()
            evaluations += collective.algorithm.evaluations
        self.evaluations = evaluations

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self):
        return self.termination_criterion.is_met

    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
            "COUNTER": self.evaluations / self.population_size,
            "COLLECTIVES": self.collectives
        }

    def get_result(self):
        return self.solutions

    def get_name(self):
        return "cMLSGA"


def incremental_stopping_condition_is_met(algo):
    eval_step = 1000
    if algo.output_path:
        if algo.termination_criterion.evaluations / (algo.output_count * eval_step) > 1:
            algo.output_job(algo.output_count * eval_step)
            algo.output_count += 1

    return algo.termination_criterion.is_met


class IncrementalcMLSGA(MultiLevelSelection):
    def __init__(self, **kwargs):
        super(IncrementalcMLSGA, self).__init__(**kwargs)
        self.output_count = 1
        self.output_path = kwargs.get("output_path", None)

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)


class IncrementalMOEAD(MOEAD):
    def __init__(self, **kwargs):
        super(IncrementalMOEAD, self).__init__(**kwargs)

        self.output_count = 1
        self.output_path = kwargs.get("output_path", None)

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)

    def update_progress(self):
        if (hasattr(self.problem, 'the_problem_has_changed') and
                self.problem.the_problem_has_changed()):
            self.solutions = self.evaluate(self.solutions)
            self.problem.clear_changed()

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.evaluations += self.offspring_population_size

    def get_observable_data(self):
        return {
            'PROBLEM': self.problem,
            'EVALUATIONS': self.evaluations,
            'SOLUTIONS': self.get_result(),
            'COMPUTING_TIME': time.time() - self.start_computing_time,
            "COUNTER": int(self.evaluations / self.population_size)
        }


def moead_options(problem, population_size, max_evaluations, evaluator, mutation_rate, crossover_rate):
    return {
        "problem": problem,
        "population_size": population_size,
        "crossover": DifferentialEvolutionCrossover(CR=crossover_rate, F=0.5, K=0.5),
        "mutation": PolynomialMutation(
            mutation_rate,
            distribution_index=20
        ),
        "aggregative_function": Tschebycheff(dimension=problem.number_of_objectives),
        "neighbor_size": 3,
        "neighbourhood_selection_probability": 0.9,
        "max_number_of_replaced_solutions": 2,
        "weight_files_path": "resources/MOEAD_weights",
        "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
        "population_evaluator": evaluator
    }


def mlsga(algorithms, problem, population_size, max_evaluations, evaluator):
    return (
        IncrementalcMLSGA,
        {
            "problem": problem,
            "population_size": population_size,
            "max_evaluations": max_evaluations,
            "number_of_collectives": 8,
            "algorithms": algorithms,
            "termination_criterion": StoppingByEvaluations(max_evaluations)
        }
    )


def nsgaii(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0):
    return (
        NSGAII,
        {
            "problem": problem,
            "population_size": population_size,
            "offspring_population_size": population_size,
            "mutation": PolynomialMutation(
                mutation_rate,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=crossover_rate, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "population_evaluator": evaluator
        }
    )


def moead(problem, population_size, max_evaluations, evaluator):
    return (
        IncrementalMOEAD,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, 1 / problem.number_of_variables, 1.0)
    )


class IncrementalOutputJob(Job):

    def __init__(self, algorithm, algorithm_tag, problem_tag, run):
        super(IncrementalOutputJob, self).__init__(algorithm, algorithm_tag, problem_tag, run)
        self.algorithm.run_tag = run

    def execute(self, output_path=""):
        self.algorithm.output_path = output_path

        super().execute(output_path)


def cmlsga(problem, population_size, max_evaluations):
    algorithm = partial(mlsga, [nsgaii, moead])

    constructor, kwargs = algorithm(problem, population_size,
                                    max_evaluations, MapEvaluator(processes=4))

    algorithm = constructor(**kwargs)
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    job = IncrementalOutputJob(
        algorithm=algorithm,
        algorithm_tag=algorithm.get_name(),
        problem_tag=problem.get_name(),
        run=3
    )

    return job
