import random
from deap import base, creator, tools


# Визначаємо функцію для максимізації
def eval_func(individual):
    x, y, z = individual
    return 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2),


# Визначаємо тип нашої популяції і її фітнес-функцію
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Ініціалізуємо популяцію та визначаємо параметри генетичного алгоритму
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Визначаємо кількість поколінь та кількість особин у популяції
num_generations = 100
population_size = 100

# Створюємо початкову популяцію та запускаємо генетичний алгоритм
population = toolbox.population(n=population_size)
for generation in range(num_generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Виводимо результат
best_individual = tools.selBest(population, k=1)[0]
print("Best individual is ", best_individual)