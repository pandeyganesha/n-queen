import random
import numpy as np
from numpy.typing import NDArray


class Solution:

    def __init__(self, chromosome, generation = None):
        self.chromosome = chromosome
        self.generation = generation

    def render_matrix(self) -> NDArray:
        """ Creates a matrix version of chessboard where 1 denotes the queen places and 0 denoted empty box"""
        n = len(self.chromosome)
        chess = np.zeros((n, n), dtype=int)
        for row in range(n):
            chess[self.chromosome[row]][row]=1
        return chess

class NQueen:

    def __init__(self, n: int, chromosome_count: int) -> None:
        self.n = self.sanitized_n(n)
        self.population = None
        self.chromosome_count = chromosome_count
        self.ans = None
        self.max_pairs = ((self.n-1)*self.n)//2
        self.populate()  # Generate random population

    def sanitized_n(self, n: int) -> int:
        if n <=3:
            raise ValueError("Value of n must be greater than 3")
        return n

    def populate(self):
        """It creates a population of random places (chromosome) for queens in each column.
        We will be taking even number for population for easy crossing over. (Won't create any differnce in working of This algo.)"""

        self.population = np.random.randint(
            0, self.n, size=(self.chromosome_count, self.n))

    def fitness(self):
        """It will calculate the fitness in percentage for each chromosome."""

        fit = []
        fit_perecent = []
        # iterate over each chromosome
        for ind, chromosome in enumerate(self.population):
            pair = 0

            # nested for loop to iterate over each value of chromosome and do comparisons with each pair possible.
            for i in range(len(chromosome)):
                for j in range(i+1, len(chromosome)):

                    # If a pair of queen is not attacking each other then + 1
                    if not self.check_attack((chromosome[i], i), (chromosome[j], j)):
                        pair += 1

            if pair == self.max_pairs:
                self.ans = Solution(self.population[ind])
                
            fit.append(pair)

        total = sum(fit)

        fit_perecent = [round((fit[i]*100/total), 2)
                        for i in range(self.chromosome_count)]
        return fit_perecent

    def check_attack(self, cord1: int, cord2: int) -> bool:
        """
        returns True if pair are attacking each other else False
        """

        # if queens in same row
        if cord1[0] == cord2[0]:
            return True

        # if queens in same diagonal
        if abs(cord1[0]-cord2[0]) == abs(cord1[1]-cord2[1]):
            return True

        return False

    def selection(self, percent: list) -> list:
        """For producing the next generation, it will create parents on the basis of fitness and probability and will also select the crossing over point."""
        selected = random.choices(
            np.arange(0, self.chromosome_count), percent, k=self.chromosome_count)
        return selected

    def crossover(self, sel: list):
        """The actual process of crossing over will be performed here."""
        new_pop = np.empty((self.chromosome_count, self.n), dtype=int)
        for i in range(0, len(sel), 2):
            # crossing of the genes at random points
            point = random.randint(1, self.n-1)

            # Used List Slicing techniques here
            new_pop[i][:point] = self.population[sel[i]][:point]
            new_pop[i+1][:point] = self.population[sel[i+1]][:point]
            new_pop[i][point:] = self.population[sel[i+1]][point:]
            new_pop[i+1][point:] = self.population[sel[i]][point:]
        self.population = new_pop

    def mutation(self):
        """"Some random mutation would be applied to the populations analogous to the evolution."""
        for chrom in self.population:
            place = random.randint(0, self.n-1)
            val = random.randint(0, self.n-1)
            chrom[place] = val  # random mutation taking place

    def solve(self, generation: int):
        """This function would be responsible for generations of the population"""
        for i in range(generation):
            percent = self.fitness()  # Then calculate the fitness of each
            if self.ans:
                self.ans.generation = i + 1
                return self.ans

            # on the basis of fitness, select randomly
            selected = self.selection(percent)
            # Do crossing over among the selected chromosomes.
            self.crossover(selected)
            self.mutation()  # Finally make some mutation
        print("Sorry! Unable to find Solution. Try increasing number of Generations or pairs of Chromosomes")


def main():
    n = int(input("Enter the number of queens to be placed : "))

    # Change the chromosome_count accordingly and keep the number even else code might not behave as expected
    chromosome_count = 500
    generation = 10000

    q = NQueen(n, chromosome_count)
    solution = q.solve(generation)
    print(f"Found solution in {solution.generation} generations!")
    print(solution.render_matrix())


if __name__ == "__main__":
    main()
