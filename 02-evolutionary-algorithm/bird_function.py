import math
import argparse
import numpy as np
import random 


def create_parser():
    """
    creates parser with optional arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iteration", type=int,
                        help='number of iterations', default=1000)
    parser.add_argument("-ps", "--population-size", type=int,
                        help='size of the population', default=20)
    parser.add_argument("-t", "--tournament", type=float,
                        help='tournament size', default=2)
    parser.add_argument("-e", "--elite", type=float,
                        help='size of the elite', default=1)
    parser.add_argument("-m", "--mutation-force", type=float,
                        help='force of the mutation (sigma)', default=0.3)
    parser.add_argument("-mp", "--mutation-probability", type=float,
                        help='probability of the mutation', default=0.5)
    # funkcja celu
    return parser


def bird_func(x, y):
    res = np.sin(x)*(np.exp(1-np.cos(y))**2)
    + np.cos(y)*(np.exp(1-np.sin(x))**2) + (x-y)**2
    return res


def easy_func(x: float, y: float) -> float:
    return x**2 + y**2


def single_crossover(parent_1, parent_2, alfa):
    """
    performs a single crossover, 
    returns two new points - children of given parents
    """
    child_1 = np.multiply(alfa, parent_1) + np.multiply((1-alfa), parent_2)
    child_2 = np.multiply(alfa, parent_2) + np.multiply((1-alfa), parent_1)
    return tuple(child_1), tuple(child_2)


def start_population(size, min, max):
    """
    generates random population in given range
    and of given size
    """
    population = []

    for _ in range(size):
        individual = (random.uniform(min, max), random.uniform(min, max))
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
    chooses random points to mutate
    """
    mutants_id = []

    for _ in range(mutant_number):
        id = random.randint(0, len(population)-1)

        while id in mutants_id:
            id = random.randint(0, mutant_number-1)
        new_x = random.gauss(population[id][0], sigma)
        new_y = random.gauss(population[id][1], sigma)
        population[id] = (new_x, new_y)
        mutants_id.append(id)

    return population



def succesion(k, population, kids):
    population_values = []
    for indv in population:
        population_values.append((indv[0], indv[1], easy_func(indv[0], indv[1])))
    
    kids_values = []
    for kid in kids:
        kids_values.append((kid[0], kid[1], easy_func(kid[0], kid[1])))
    
    population_values.sort(key=lambda x: x[2])
    elite = population_values[:k]
    
    next_population = elite + kids_values
    next_population.sort(key=lambda x: x[2])
    next_population = next_population[:-k]

    return next_population


def algortithm():
    population = start_population(10, 10, 10)

    for _ in range(500):
        print(_)

        tour = tournament_reproduction(2, population, easy_func)

        kids = crossover(tour, 0.1)
        kids = mutation(kids, 3, 0.1)

        population = succesion(1, population, kids)
        print(population[0])


algortithm()


