""" Hotspot definition """
import numpy as np

class Hotspot():
	def __init__(self, code, value, visit_cost, latitude=None, longitude=None, travel_cost=None, group="", size=''):
		self.code = code
		self.value = value
		self.visit_cost = visit_cost
		self.latitude = latitude
		self.longitude = longitude
		if travel_cost:
			self.travel_cost = travel_cost
		else:
			self.travel_cost = {}
		self.group = group
		self.size = size

	def __str__(self):
		return str(self.code)

	def __eq__(self, hotspot):
		return self.code == hotspot.code

	def get_travel_cost(self, code):
		return self.travel_cost[code]

	def set_travel_cost(self, code, cost):
		self.travel_cost[code] = cost

	def get_visit_cost(self):
		return self.visit_cost

	def get_code(self):
		return self.code

	def get_value(self):
		return self.value

	def get_euclidean_distance(self, hotspot):
		summand_a = np.square(self.latitude - hotspot.latitude)
		summand_b = np.square(self.longitude - hotspot.longitude)
		return np.sqrt(summand_a+summand_b)

	def verbose(self):
		out = "Code: " + str(self.code) + "\n"
		out += "Value: " + str(self.value) + "\n"
		out += "Visit Cost: " + str(self.visit_cost) + "\n"
		out += "Travel Cost: " + str(self.travel_cost)
		return out