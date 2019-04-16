"""
	Planner for assigning pdvs to GPVs

	Strategy:
	- Genetic Algorithms. Solutions are represented as individuals where each chromosome is a
		salepoint and the order of the genome in the chromosome is the order of visit until the
		maximum capacity is reached.

	Important parameters:
	- Generation size
	- Number of iterations/Stop criteria
	- Mutation probability
	- Crossover probability

	Author: Lauro Bravar
"""
import csv
import argparse
import random as rdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from set_up import random_init, build_distance_matrix, hotspots_to_file

def initialize(number_of_chromosomes, pdvs):
	""" Initialize a generation with random individuals """
	generation = np.empty([number_of_chromosomes, len(pdvs)]).astype(int)
	chromosome = np.array([]).astype(int)
	for i in range(len(pdvs)):
		chromosome = np.append([chromosome], [i])
	for i in range(number_of_chromosomes):
		np.random.shuffle(chromosome)
		generation[i] = chromosome
	return generation

def selection(ranking):
	""" Evolutionary selection function. Uses weighted rankng probability. """
	number_of_indiv = len(ranking)
	selected = []
	rank = 0
	ranges = [weighted_rank_prob(number_of_indiv, 0), weighted_rank_prob(number_of_indiv, number_of_indiv-1)]
	while len(selected) != number_of_indiv:
		if weighted_rank_prob(number_of_indiv, rank) >= rdm.uniform(ranges[0], ranges[1]):
			selected.append(ranking[rank])
		rank = (rank+1)%number_of_indiv
	return selected

def weighted_rank_prob(size, rank):
	""" Weighted ranking probability calculator """
	numerator = size - rank - 1
	denominator = size*(size+1)/2
	prob = numerator/denominator
	return prob

def get_repeated(elite):
	""" Get the indexes of repeated chromosomes """
	rep_indices = []
	elite = elite.tolist()
	for chrom in elite:
		indices = [index for index, value in enumerate(elite) if value == chrom]
		if len(indices) > 1:
			for index in indices[1:]:
				rep_indices.append(index)
	return rep_indices

def reproduce(elite, indices):
	"""
	Evolutionary reproduction function. Crosses over certain individuals according to a probability
	"""
	elite = elite.tolist()
	new_generation = []
	for index in indices:
		while True:
			mut_index = rdm.randint(0, len(elite) - 1)
			if elite[mut_index] != elite[index]:
				break
		offspring1, offspring2 = cycle_crossover(elite[index], elite[mut_index])
		new_generation.append(offspring1)
		new_generation.append(offspring2)
		if len(new_generation) >= len(elite):
			break
	return new_generation

def cycle_crossover(parent1, parent2):
	""" Returns crossedover offsprings """
	# Choose a genome randomly to mutate
	position = rdm.randint(0, len(parent1)-1)

	# Init the positions array
	positions = [position, position]

	# Crossover
	while len(positions) > 1:
		if positions[0] == position:
			position = positions[1]
		else:
			position = positions[0]
		aux_genome = parent1[position]
		parent1[position] = parent2[position]
		parent2[position] = aux_genome
		positions, = np.where(parent1 == parent1[position])

	# Parents are now offsprings
	return parent1, parent2

def mutate(chromosome):
	""" Returns a mutated chromosome """
	# Choose two genomes
	position1 = rdm.randint(0, len(chromosome)-1)
	position2 = rdm.randint(0, len(chromosome)-1)
	# Swap it with another genome
	aux_genome = chromosome[position1]
	chromosome[position1] = chromosome[position2]
	chromosome[position2] = aux_genome
	return chromosome

def fitness(chromosome, pdvs, max_cap, output=False):
	""" Returns the fitness of the chromosome """
	cost = pdvs[chromosome[0]].get_visit_cost()
	# If just visiting the first pdv exceeds the max cap
	if cost > max_cap:
		if output:
			return 0, cost, 0
		return 0
	value = pdvs[chromosome[0]].get_value()
	for i in range(len(chromosome)-1):
		cost_aux = cost + pdvs[chromosome[i]].get_travel_cost(pdvs[chromosome[i+1]].get_code()) +\
																				pdvs[chromosome[i+1]].get_visit_cost()
		if cost_aux > max_cap:
			if output:
				return value, cost, i
			return value
		else:
			cost = cost_aux
			value += pdvs[chromosome[i]].get_value()
	return value

def print_generation(generation, pdvs, max_cap):
	""" Print a generation """
	for chromosome in generation:
		print("Fitness: " + str(fitness(chromosome, pdvs, max_cap, output=True)[0]))
		print("Cost: " + str(fitness(chromosome, pdvs, max_cap, output=True)[1]))
		print("Length: " + str(fitness(chromosome, pdvs, max_cap, output=True)[2]))

def plot_solution(pdvs, solution, score, cost):
	""" Plot the solution """
	# Size and opacity of the instance points in scatter
	size = 40
	opacity = 0.1
	patches = []

	# Color map and norm
	cmap = plt.cm.magma
	norm = plt.Normalize(0, len(solution))

	# X and Y coordinates of the instances
	pointsx = np.array([pdv.latitude for pdv in pdvs])
	pointsy = np.array([pdv.longitude for pdv in pdvs])

	plt.scatter(pointsx, pointsy, size, c=np.ones(len(pointsx)).astype(int), cmap=cmap, norm=norm,
													alpha=opacity, marker="s")

	# X and Y coordinates of the solution
	solutionsx = np.array([pdvs[index].latitude for index in solution])
	solutionsy = np.array([pdvs[index].longitude for index in solution])

	plt.plot(solutionsx, solutionsy, color=cmap(len(solution)/(len(solution)+1)), marker=".")

	# Add label
	patch = mpatches.Patch(color=cmap(len(solution)/(len(solution)+1)), label="Score " + str(score)[:6])
	patches.append(patch)

	patch = mpatches.Patch(color=cmap(len(solution)/(len(solution)+1)), label="Cost " + str(cost)[:6])
	patches.append(patch)

	# Labels & Legend
	plt.title("Solution")
	plt.xlabel("Latitude")
	plt.ylabel("Longitud")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=patches)
	plt.show()

def parse_args():
	""" Parse arguments from command line. Return the arguments in a dict """
	parser = argparse.ArgumentParser()
	parser.add_argument("generation_size", help="Generation size", type=int)
	parser.add_argument("iterations", help="Number of iterations per solution", type=int)
	parser.add_argument("mutation_prob", help="Mutation probability", type=float)
	parser.add_argument("crossover_prob", help="Crossover probability", type=float)
	parser.add_argument("max_cap", help="Maximum availability in hours", type=int)
	parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")
	parser.add_argument("-s", "--show", help="Enable graph and image displaying", action="store_true")

	args = vars(parser.parse_args())
	return args

def main():
	""" Main flow """
	# Parse the arguments of the program
	args = parse_args()

	generation_size = args["generation_size"]
	iterations = args["iterations"]
	mutation_prob = args["mutation_prob"]
	crossover_prob = args["crossover_prob"]
	print_show = [args["verbose"], args["show"]]
	max_cap = args["max_cap"]
	
	hotspots = random_init(100, positions=True)
	build_distance_matrix(hotspots)
	hotspots_to_file(hotspots, "prueba", "C:/Users/Lauro/Desktop/genetic/genetic-optimizer/test_files/")
	generation = initialize(generation_size, hotspots)
	fittest_value = 0
	fittest = []
	for i in range(iterations):
		print(str(i*100/iterations)[:5] + "%")
		print("GENSIZE = " + str(len(generation)))
		fitness_array = np.array([fitness(chromosome, hotspots, max_cap) for chromosome in generation])
		sorted_fitness_array = np.argsort(fitness_array)[::-1]
		parents_indices = selection(sorted_fitness_array)
		if fittest_value < fitness_array[sorted_fitness_array[0]]:
			fittest_value = fitness_array[sorted_fitness_array[0]]
			fittest = generation[sorted_fitness_array[0]]
		parents = np.array([generation[index] for index in parents_indices])
		for chromosome in parents:
			if mutation_prob > rdm.random():
				chromosome = mutate(chromosome)
		indices_to_cross = get_repeated(parents)
		generation = reproduce(parents, indices_to_cross)
		if print_show[0]:
			print("Fittest Value: " + str(fittest_value))

	value, cost, length = fitness(fittest, hotspots, max_cap, output=True)
	solution = fittest[0:(length-1)]
	if print_show[0]:
		print("Solution: ")
		print(solution)
		print("Cost of solution: ")
		print(cost)
		print("Value of solution: ")
		print(value)
	visited_markets = [hotspots[solution_].get_code() for solution_ in solution]
	if print_show[1]:
		plot_solution(hotspots, solution, value, cost)


if __name__ == '__main__':
	main()
