import sys
from datetime import datetime
import time

class Logger(object):
    def __init__(self, fn):
        self.terminal = sys.stdout
        self.log = open(fn, "a")
        # Textual month, day and year
        self.st = time.time()
        now = datetime.now()
        self.write("started: "+ now.strftime("%d/%m/%Y %H:%M:%S")+'\n')
        self.write('---------------------------------\n')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def __del__(self):
        self.write('---------------------------------\n')
        now = datetime.now()
        self.write("end at: " + now.strftime("%d/%m/%Y %H:%M:%S")+'\n')
        et = time.time()
        elapsed_time = (et - self.st)/60
        hours = elapsed_time // 60
        minutes = elapsed_time % 60
        self.write("execution time: " + str(hours) + ' hours, ' + str(minutes) + ' minutes \n')
        self.write('\n')
