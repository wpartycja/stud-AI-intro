import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm


def bird_func(x, y):
    res = np.sin(x)*(np.exp(1-np.cos(y))**2)
    + np.cos(y)*(np.exp(1-np.sin(x))**2) + (x-y)**2
    return res


def easy_func(x: float, y: float) -> float:
    return x**2 + y**2


def rosenbrock_func(x, y):
    return (1-x)**2 + 100 * (y-x**2)**2


def single_crossover(parent_1, parent_2, alfa):
    """
    performs a single crossover,
    returns two new points - children of given parents
    """
    child_1 = np.multiply(alfa, parent_1) + np.multiply((1-alfa), parent_2)
    child_2 = np.multiply(alfa, parent_2) + np.multiply((1-alfa), parent_1)
    return tuple(child_1), tuple(child_2)


def start_population(size, min_x, max_x, min_y, max_y):
    """
    generates random population in given range
    and of given size
    """
    population = []

    for _ in range(size):
        individual = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        population.append(individual)

    return population


def tournament_reproduction(tournament_size, population, func):
    """
    performs a tournament reproduction
    tournament_size: number of competitors in one tournament
    func: aim function
    returns winner of every tournament
    """
    new_parents = []

    for _ in range(len(population)-1):
        # picking competitors to one round
        competitors = random.choices(population, k=tournament_size)
        # assuming that first one is the winner
        winner = (competitors[0][0], competitors[0][1])

        # checking who is the real winner
        for competitor in competitors:
            if competitor == winner:
                continue
            x, y = competitor[0], competitor[1]

            if func(x, y) < func(winner[0], winner[1]):
                competitor = winner

        new_parents.append(winner)

    return new_parents


def crossover(population, alfa):
    """
    performs crossover on the whole population
    returns new generaration
    """
    youngsters = []

    for _ in range(len(population)//2+1):
        parents = random.choices(population, k=2)
        children = single_crossover(parents[0], parents[1], alfa)
        youngsters.extend(children)

    return youngsters


def mutation(population, mutant_number, sigma):
    """
    chooses random points to mutate,
    returns population with muted individuals
    """
    mutants_id = []

    for _ in range(mutant_number):
        id = random.randint(0, len(population)-1)

        # checking if we do not mute the same individual two times
        while id in mutants_id:
            id = random.randint(0, mutant_number-1)

        new_x = random.gauss(population[id][0], sigma)
        new_y = random.gauss(population[id][1], sigma)
        population[id] = (new_x, new_y)
        mutants_id.append(id)

    return population


def elite_succesion(k, population, kids, func):
    """
    choses k best individuals from population after selection and from children,
    then it joins with the children group and cuts k worst individuals
    returns new generation
    """
    population_values = []
    for indv in population:
        population_values.append((indv[0], indv[1], func(indv[0], indv[1])))

    kids_values = []
    for kid in kids:
        kids_values.append((kid[0], kid[1], func(kid[0], kid[1])))

    population_values.sort(key=lambda x: x[2])
    elite = population_values[:k]

    next_population = elite + kids_values
    next_population.sort(key=lambda x: x[2])
    next_population = next_population[:-k]

    return next_population


if __name__ == "__main__":
    # creating random population
    # population size, min_x, max_x, min_y, max_y
    population = start_population(20, 1, 2, 1, 2)
    func = rosenbrock_func
    # data for the plot
    first_x = [indv[0] for indv in population]
    first_y = [indv[1] for indv in population]
    first_values = [func(indv[0], indv[1]) for indv in population]

    all_x, all_y, all_values = [], [], []

    # starting evolution!
    # number of iterations
    for _ in range(500):
        print(_)

        # selection
        # size of tournament, population, aim func
        selected_indv = tournament_reproduction(2, population, func)

        # crossover
        # selected individuals, alfa
        kids = crossover(selected_indv, 0.1)

        # mutation
        # list with individuals, number of indviduals to mutate, sigma
        kids = mutation(kids, 3, 0.1)

        # succesion
        # size of elite, population, kids, aim func
        population = elite_succesion(1, population, kids, func)

        # helps to follow the algorithm
        print(population[0])

        # data for the plot
        all_x.extend(indv[0] for indv in population)
        all_y.extend(indv[1] for indv in population)
        all_values.extend(func(indv[0], indv[1]) for indv in population)

    # creating a plot
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)

    x, y = np.meshgrid(x, y)
    values = func(x, y)

    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.scatter(first_x, first_y, first_values, c='red')
    axis.scatter(all_x, all_y, all_values, c='green')
    axis.contour3D(x, y, values)
    axis.set_title('Function')
    axis.plot_wireframe(x, y, values, rstride=7, cstride=7, color='black')

    axis.view_init(elev=90, azim=42)
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_zlabel('z')
    plt.contour(x, y, values)
    plt.title("Function")
    plt.show()
