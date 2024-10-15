import random
import numpy as np

class GeneticAlgorithm:
    VOCAB = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '

    def __init__(
            self, 
            target, 
            population_size=100, 
            mutation_rate=0.01,
            generations=1000
        ) -> None:
        
        self.target = target
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = None

    def initialize_population(self):
        
        # random strings of the same length as the target
        self.population = [
            ''.join(
                random.choice(self.VOCAB)
                for _ in range(len(self.target))
            ) 
            for _ in range(self.population_size)
        ]

    def fitness(self, individual):
        
        # fitness(h) = correct(h) ^ 2
        return sum(
            1 
            for c1, c2 in zip(individual, self.target)
            if c1 == c2
        ) ** 2

    def selection(self):
        
        # select by random weighted sampling
        fitness_sum = sum(
            self.fitness(individual)
            for individual in self.population
        )

        parents = np.random.choice(
            self.population,
            size=2,
            replace=False,
            p=[
                self.fitness(individual) / fitness_sum
                for individual in self.population
            ]
        )

        return parents

    def crossover(self, parent1, parent2):
        
        # single point crossover
        pt = random.randint(0, len(self.target))
        offspring1 = parent1[:pt] + parent2[pt:]
        offspring2 = parent2[:pt] + parent1[pt:]

        return offspring1, offspring2

    def mutation(self, offspring):
        
        # random character mutation
        return ''.join(
            random.choice(self.VOCAB)
            if random.random() < self.mutation_rate
            else c
            for c in offspring
        )

    def evolve(self):
        
        self.initialize_population()
        best_individual = max(
            self.population,
            key=self.fitness
        )
        print(f'Gen 0: {best_individual}, fitness: {self.fitness(best_individual)}')

        for gen in range(self.generations):

            new_population = []

            for _ in range(self.population_size // 2):

                parent1, parent2 = self.selection()
                offspring1, offspring2 = self.crossover(parent1, parent2)

                offspring1 = self.mutation(offspring1)
                offspring2 = self.mutation(offspring2)

                new_population.extend([offspring1, offspring2])

            self.population = new_population

            best_individual = max(
                self.population,
                key=self.fitness
            )

            print(f'Gen {gen}: {best_individual}, fitness: {self.fitness(best_individual)}')

            if best_individual == self.target:
                break
        
    def run(self):
        
        self.evolve()
        best_individual = max(
            self.population,
            key=self.fitness
        )
        print(f'Best individual: {best_individual}')
        print(f'Fitness: {self.fitness(best_individual)}')