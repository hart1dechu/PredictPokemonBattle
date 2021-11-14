import csv
import math
from functools import reduce
import operator
from numpy import sqrt, true_divide
import random
import numpy as np
from sklearn.model_selection import KFold

def split_lines(input, seed, output1, output2):
  random.seed(seed)
  output1 = open(output1,'a')
  output1.truncate(0)
  output2 = open(output2,'a')
  output2.truncate(0)
  for line in open(input, 'r').readlines(): 
      if (random.random() < 0.5):
          write = output1;
      else:
          write = output2;
      write.write(line);

def read_data(filename):
  X = []
  Y = []
  with open(filename) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for line in csv_reader:
          X.append(list(map(float,line[:1])))
          Y.append(line[2] == line[0])

  return (X,Y);