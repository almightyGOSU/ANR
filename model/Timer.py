import time
from datetime import datetime



def getMinutes(elapsedTime):

	return elapsedTime / 60

def getHoursMinutes(elapsedTime):

	hours = (elapsedTime // 60 // 60)
	minutes = (elapsedTime // 60) - (hours * 60)
	seconds = elapsedTime - (hours * 60 * 60) - (minutes * 60)
	return int(hours), int(minutes), int(seconds)


class Timer():

	def __init__(self):

		# Used for storing "keys"
		# E.g. "global" can be used to measure total program execution time, "init" for initialization, etc
		self.timer_dict = {}


	def startTimer(self, key = "global"):

		self.timer_dict[key] = time.time()


	def getElapsedTime(self, key = "global"):

		if key not in self.timer_dict.keys():
			return 0

		return time.time() - self.timer_dict[key]


	def getElapsedTimeStr(self, key = "global", conv2Mins = False, conv2HrsMins = False):

		elapsedTime = self.getElapsedTime(key = key)

		if conv2HrsMins:
			hours, minutes, seconds = getHoursMinutes(elapsedTime)
			return "Elapsed Time: {:,.2f}s ({:d}:{:02d}:{:02d})".format( elapsedTime, hours, minutes, seconds )

		elif conv2Mins:
			minutes = getMinutes(elapsedTime)
			return "Elapsed Time: {:,.2f}s ({:.2f} {})".format( elapsedTime, minutes, "minutes" if minutes > 1 else "minute" ) 

		else:
			return "Elapsed Time: {:,.2f}s".format( elapsedTime )



if __name__ == '__main__':

	elapsedTime = 10
	hours, minutes, seconds = getHoursMinutes(elapsedTime)
	print("Elapsed Time: {:,.2f}s ({:d}:{:02d}:{:02d})".format( elapsedTime, hours, minutes, seconds ))

	elapsedTime = 12.89
	hours, minutes, seconds = getHoursMinutes(elapsedTime)
	print("Elapsed Time: {:,.2f}s ({:d}:{:02d}:{:02d})".format( elapsedTime, hours, minutes, seconds ))

	elapsedTime = 75
	hours, minutes, seconds = getHoursMinutes(elapsedTime)
	print("Elapsed Time: {:,.2f}s ({:d}:{:02d}:{:02d})".format( elapsedTime, hours, minutes, seconds ))

	elapsedTime = 125
	hours, minutes, seconds = getHoursMinutes(elapsedTime)
	print("Elapsed Time: {:,.2f}s ({:d}:{:02d}:{:02d})".format( elapsedTime, hours, minutes, seconds ))

	elapsedTime = 1800
	hours, minutes, seconds = getHoursMinutes(elapsedTime)
	print("Elapsed Time: {:,.2f}s ({:d}:{:02d}:{:02d})".format( elapsedTime, hours, minutes, seconds ))

	elapsedTime = 3675
	hours, minutes, seconds = getHoursMinutes(elapsedTime)
	print("Elapsed Time: {:,.2f}s ({:d}:{:02d}:{:02d})".format( elapsedTime, hours, minutes, seconds ))

	elapsedTime = 7338.925
	hours, minutes, seconds = getHoursMinutes(elapsedTime)
	print("Elapsed Time: {:,.2f}s ({:d}:{:02d}:{:02d})".format( elapsedTime, hours, minutes, seconds ))

