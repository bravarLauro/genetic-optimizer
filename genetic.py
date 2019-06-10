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
from set_up import random_init, build_distance_matrix, hotspots_to_file, build_test_case

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
	selected_ranks = []
	rank = 0
	weighted_ranks = [weighted_rank_prob(number_of_indiv, i) for i in range(number_of_indiv)]
	while len(selected) != number_of_indiv/2:
		if weighted_rank_prob(number_of_indiv, rank) >= rdm.uniform(0, max(weighted_ranks)) and rank not in selected_ranks:
			selected.append(ranking[rank])
			selected_ranks.append(rank)
			weighted_ranks[rank] = 0
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

def reproduce(elite, crossover_prob):
	"""
	Evolutionary reproduction function. Crosses over certain individuals according to a probability
	"""
	elite = elite.tolist()
	new_generation = []
	while len(new_generation) < len(elite)*2:
		for index, chrom in enumerate(elite):
			if crossover_prob > rdm.random():
				other_index = rdm.randint(0, len(elite) - 1)
				if other_index == index:
					if other_index == 0:
						other_index += 1
					else:
						other_index -= 1
				offspring1, offspring2 = cycle_crossover(elite[index], elite[other_index])
				new_generation.append(offspring1)
				new_generation.append(offspring2)
			else:
				new_generation.append(chrom)
			if len(new_generation) >= len(elite)*2:
				break
	return new_generation[:len(elite)*2]

def cycle_crossover(parent1, parent2):
	""" Cycle crossover the two parents """
	offspring1 = [-1 for _ in parent1]
	offspring2 = [-1 for _ in parent1]
	position = rdm.randint(0, len(parent1)-1)
	positions_checked = []
	while position not in positions_checked:
		positions_checked.append(position)
		offspring1[position] = parent1[position]
		offspring2[position] = parent2[position]
		position = parent1.index(parent2[position])
	for index, element in enumerate(parent1):
		if index not in positions_checked:
			offspring1[index] = parent2[index]
			offspring2[index] = element
	return offspring1, offspring2

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
	parser.add_argument("number_of_points", help="Number of points to randomly initialize", type=int)
	parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")
	parser.add_argument("-s", "--show", help="Enable graph and image displaying", action="store_true")
	parser.add_argument("-c", "--case", help="Hotspot case to use")

	args = vars(parser.parse_args())
	return args

def plot_evolution(iterations, training_fittest, generation_size, mutation_prob, crossover_prob, number_of_points):
	plt.plot(range(0, iterations, 10), training_fittest)
	plt.title("GS: " + str(generation_size) + ", MP: " + str(mutation_prob) + ", CP: " +\
										 str(crossover_prob) + ", NP: " + str(number_of_points))
	plt.xlabel("Number of iterations")
	plt.ylabel("Score")
	# plt.show()
	plt.savefig('test_results/' + str(generation_size) + '_' + str(iterations) + '_' +\
							str(mutation_prob) + '_' + str(crossover_prob) + '_' +\
							str(number_of_points) + '.png')

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
	number_of_points = args["number_of_points"]
	if args['case']:
		hotspots = build_test_case(args['case'], number_of_points)
	else:
		hotspots = random_init(number_of_points, positions=True)
		build_distance_matrix(hotspots)
		hotspots_to_file(hotspots, "prueba_" + str(number_of_points), "C:/Users/Lauro/Desktop/genetic/genetic-optimizer/test_files/")
	generation = initialize(generation_size, hotspots)
	fittest_value = 0
	fittest = []
	training_fittest = []
	for i in range(iterations):
		print(str(i*100/iterations)[:5] + "%")
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
		generation = reproduce(parents, crossover_prob)
		if print_show[0]:
			print("Fittest Value: " + str(fittest_value))
		if i % 10 == 0:
			training_fittest.append(fittest_value)

	plot_evolution(iterations, training_fittest, generation_size, mutation_prob, crossover_prob, number_of_points)

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
