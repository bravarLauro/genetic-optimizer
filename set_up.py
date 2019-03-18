""" Random set up initializer """
from classes.hotspot import Hotspot
import numpy as np

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

def build_distance_matrix(hotspots):
	for hotspot_ii in hotspots:
		for hotspot_jj in hotspots:
			distance = hotspot_ii.get_euclidean_distance(hotspot_jj)
			hotspot_ii.set_travel_cost(hotspot_jj.code, distance)
