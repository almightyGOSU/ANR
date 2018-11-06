from .utilities import *

import os
import sys
import re



# Creating a folder
def mkdir_p(path):

	if path == "":
		return
	try:
		os.makedirs(path)
	except:
		pass


# Saving all the arguments
def print_args(args, path = None):

	if path:
		output_file = open(path, "w")

	args.command = " ".join(sys.argv)
	args.command = re.sub(r"-flood [0-1] ", "", args.command)
	args.command = re.sub(r"-verbose [0-9] ", "", args.command)
	args.command = args.command.replace("PyTorchTEST.py ", "")

	items = vars(args)
	if path:
		output_file.write("{}\n".format(TEXT_SEP))
	for key in sorted(items.keys(), key = lambda s: s.lower()):
		value = items[key]
		if not value:
			value = "None"
		if path is not None:
			output_file.write("  " + key + ": " + str(items[key]) + "\n")
	if path:
		output_file.write("{}\n".format(TEXT_SEP))
		output_file.close()

	print("\nCommand: {}".format( args.command ))
	del args.command


# Helper class for logging everything to "logs.txt"
class Logger():

	def __init__(self, out_dir, log_path, args):

		self.out_dir = out_dir
		self.log_path = log_path

		mkdir_p(self.out_dir)
		with open(self.log_path, 'w+') as f:
			f.write("")
		print_args(args, path = self.log_path)


	def log(self, txt, log_path = None, print_txt = True):

		if(log_path is None):
			with open(self.log_path, 'a+') as f:
				f.write(txt + "\n")
		else:
			with open(log_path, 'a+') as f:
				f.write(txt + "\n")

		if(print_txt):
			print(txt)


	def logQ(self, txt, log_path = None):

		self.log(txt, log_path = log_path, print_txt = False)


	def emptyFile(self, log_path = None):
		with open(log_path, 'w+') as f:
			f.write("")


