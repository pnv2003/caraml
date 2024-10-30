from models.evolve import GeneticAlgorithm

str = "Hello World"
population_size = 100
mutation_rate = 0.01
generations = 1000

ga = GeneticAlgorithm(str, population_size, mutation_rate, generations)
ga.run()
