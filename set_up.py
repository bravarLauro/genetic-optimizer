""" Random set up initializer """
from classes.hotspot import Hotspot
import numpy as np
import csv

def random_init(length, values_max=100, visit_cost_max=100, positions=False, canvas_max=100, travel_costs=False, travel_cost_max=100, groups=False, sizes=False):
	hotspots = []
	for code in range(length):
		value = np.random.randint(0, values_max)
		visit_cost = np.random.randint(0, visit_cost_max)
		hotspot = Hotspot(code, value, visit_cost)

		if positions:
			latitude = np.random.randint(0, canvas_max)
			longitude = np.random.randint(0, canvas_max)
			hotspot.latitude = latitude
			hotspot.longitude = longitude

		hotspots.append(hotspot)
	return hotspots

def hotspots_to_file(hotspots, name, path):
	hs_file = open(path + name + "_hotspots.csv", "w+", newline="")
	distances_file = open(path + name + "_distances.csv", "w+", newline="")
	hs_writer = csv.writer(hs_file, delimiter=";")
	distances_writer = csv.writer(distances_file, delimiter=";")
	distances_writer.writerow([hotspot.code for hotspot in hotspots])
	for hotspot in hotspots:
		hs_writer.writerow([hotspot.code, hotspot.value, hotspot.visit_cost])
		distances_writer.writerow([hotspot.travel_cost[hotspot_jj.code] for hotspot_jj in hotspots])
	hs_file.close()
	distances_file.close()


def build_distance_matrix(hotspots):
	for hotspot_ii in hotspots:
		for hotspot_jj in hotspots:
			distance = hotspot_ii.get_euclidean_distance(hotspot_jj)
			hotspot_ii.set_travel_cost(hotspot_jj.code, distance)

def build_test_case(case_name, length):
	hotspots_file = open(case_name + "_hotspots.csv")
	distances_file = open(case_name + "_distances.csv")
	hotspots_reader = csv.reader(hotspots_file, delimiter=";")
	distances_reader = csv.reader(distances_file, delimiter=";")
	codes = next(distances_reader)
	hotspots = []
	for _ in range(length):
		hotspots_line = next(hotspots_reader)
		distances_line = next(distances_reader)
		hotspot = Hotspot(hotspots_line[0], int(hotspots_line[1]), float(hotspots_line[2]))
		for index in range(length):
			hotspot.set_travel_cost(codes[index], float(distances_line[index]))
		hotspots.append(hotspot)
	return hotspots

if __name__ == "__main__":
	hotspots = build_test_case("./test_files/test_case1", 3)
	for hotspot in hotspots:
		print(hotspot.verbose())

