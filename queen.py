import random
import numpy as np
from visual_queen import matrix_render


class ExitLoop(Exception):
    """
    Raises:
        ExitLoop """
    pass


class Solution:

    def __init__(self, solution_array):
        self.solution_array = solution_array

    def matrix_render(self):
        """ Creates a matrix version of chessboard where 1 denotes the queen places and 0 denoted empty box"""
        n = len(self.solution_array)
        chess = np.zeros((n, n), dtype=int)
        for row in range(n):
            chess[self.solution_array[row]][row]=1
        return chess

class NQueen:
    def __init__(self, n, chromosome_count) -> None:
        self.n = self.sanitized_n(n)
        self.population = None
        self.chromosome_count = chromosome_count
        self.ans = None

    def sanitized_n(self, n):
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
        mxm = ((self.n-1)*self.n)//2  # maximum number of non attacking pairs possible
        fit = []
        fit_perecent = []
        # iterate over each chromosome
        for ind1, pop in enumerate(self.population):
            pair = 0

            # nested for loop to iterate over each value of chromosome and do comparisons with each pair possible.
            for i in range(len(pop)):
                for j in range(i+1, len(pop)):

                    # If a pair of queen is not attacking each other then + 1
                    if not self.check_attack((pop[i], i), (pop[j], j)):
                        pair += 1

            if pair == mxm:
                self.ans = Solution(self.population[ind1])
                raise ExitLoop
                
            fit.append(pair)

        total = sum(fit)

        fit_perecent = [round((fit[i]*100/total), 2)
                        for i in range(self.chromosome_count)]
        return fit_perecent

    def check_attack(self, cord1, cord2):
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

    def selection(self, percent):
        """For producing the next generation, it will create parents on the basis of fitness and probability and will also select the crossing over point."""
        selected = random.choices(
            np.arange(0, self.chromosome_count), percent, k=self.chromosome_count)
        return selected

    def crossover(self, sel):
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

    def solve(self, generation):
        """This function would be responsible for generations of the population"""
        self.populate()  # Generate random population
        for i in range(generation):
            try:
                percent = self.fitness()  # Then calculate the fitness of each
            except ExitLoop:  # if an error occurs , it stops the generation showing that it has got the best solution
                print(str(i)+"th generation resulted : ", self.ans.solution_array)
                print(self.ans.matrix_render())
                exit()
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


if __name__ == "__main__":
    main()
