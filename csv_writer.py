import os
import csv
class Detected:
    def __init__(self, header=['ID', 'name', 'frame', 'conf', 'max_speed']):
        self.header = header
        self.path = os.getcwd()
        self.path += r"\results\scores.csv"

        self.writer = csv.writer(open(self.path, 'w'))
        self.writer.writerow(header)
    def save(self, list):
        self.writer.writerow(list)

