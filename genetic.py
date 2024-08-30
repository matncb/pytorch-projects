import random

length = 10 #length = tamanho do genoma
size = 10 #size = tamanho da populacao
num = 1 #num = numero de mutacoes
probability = 0.5  #probability = probabilidade de mutações
generation_limit = 50
max_fitness = 10

def generate_genome():
    return random.choices([0,1], k = length)

def generate_population():
    return [generate_genome() for i in range(size)]

def fitness(genome):
    return 0

def select_pair(population):
    return random.choices(
        population = population,
        weights = [fitness(genome) for genome in population ],
        k = 2
    )

def single_point_crossover(genome1, genome2):
    p = random.randint(1, length -1)

    if length < 2:
        return genome1, genome2
    else:
        return (genome1[0:p] + genome2[p:], genome2[0:p] + genome1[p:])

def mutation(genome):
    for i in range(num):
        index = random.randrange(length)
        if random.random() < probability:
            #alterar valor
            # genome = abs(genome[index] - 1)
            pass
    return genome

def run_evolution():
    population = generate_population()

    for i in range(generation_limit):
        population = sorted(
            population,
            key = lambda genome: fitness(genome),
            reverse = True
        )

        if fitness(population[0]) >= max_fitness:
            break

        next_generation = population[0:2] #manter top 2

        for j in range(int(size/2) - 1):
            parents = select_pair(population)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a)
            offspring_b = mutation(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
