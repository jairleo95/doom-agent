import json

class ParamsManager(object):

	def __init__(self, params_file):
		self.params = json.load(open(params_file, 'r'))#r de read


	def get_params(self):
		return self.params

	def get_agent_params(self):
		return self.params['agent']

	def get_environment_params(self):
		return self.params['environment']

	def update_agent_params(self, **kwargs):
		for key, value in kwargs.items():
			if key in self.get_agent_params().keys():
				self.params['agent'][key] = value

	def export_agent_params(self, file_name):
		with open(file_name, 'w') as f:
			json.dump(self.params['agent'], f, indent = 4, separators = (',', ':'), sort_keys = True)
			f.write("\n")

	def export_environment_params(self, file_name):
		with open(file_name, 'w') as f:
			json.dump(self.params['environment'], f, indent = 4, separators = (',', ':'), sort_keys = True)
			f.write("\n")