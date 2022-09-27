class LinearDecaySchedule(object):

	def __init__(self, initial_value, final_value, max_steps):
		assert initial_value > final_value, "El valor inicial debe ser estrictamente mayor que el valor final."
		self.initial_value = initial_value
		self.final_value = final_value
		self.decay_factor = (initial_value - final_value)/ max_steps

	def __call__(self, step_num):
		current_value = self.initial_value - step_num * self.decay_factor
		if current_value < self.final_value:
			current_value = self.final_value
		return current_value

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	epsilon_initial = 1.0
	epsilon_final = 0.01
	MAX_NUM_EPISODES = 100
	STEPS_PER_EPISODE = 100
	max_steps = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE
	linear_schedule = LinearDecaySchedule(initial_value = epsilon_initial,
                                                        final_value = epsilon_final,
                                                        max_steps = max_steps)
	print("max_steps:"+str(max_steps))

	epsilons = [linear_schedule(step) for step in range(MAX_NUM_EPISODES * STEPS_PER_EPISODE)]
	plt.plot(epsilons)
	plt.show()