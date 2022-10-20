from matplotlib import pyplot as plt


def print_solution(my_algo, pop_evolved, params):
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
