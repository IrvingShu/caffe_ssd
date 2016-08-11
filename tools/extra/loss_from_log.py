#!/usr/bin/env python
# Fan Yang, 2016/08/11

from __future__ import print_function
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

from utils import strstr

def main():
  log_files = process_arguments(sys.argv)

  train_iteration = []
  train_loss      = []
  test_iteration  = []
  detection_eval  = []

  base_test_iter  = 0
  base_train_iter = 0

  for log_file in log_files:
    with open(log_file, 'rb') as f:
      if len(train_iteration) != 0:
        base_train_iter = train_iteration[-1]
        base_test_iter = test_iteration[-1]

      for line in f:
        # TRAIN NET
        if strstr(line, 'Iteration') and strstr(line, 'lr'):
          matched = match_iteration(line)
          train_iteration.append(int(matched.group(1))+base_train_iter)

        elif strstr(line, 'Train net output'):
          matched = match_loss(line)
          train_loss.append(float(matched.group(1)))

        # TEST NET
        elif strstr(line, 'Testing net'):
          matched = match_iteration(line)
          test_iteration.append(int(matched.group(1))+base_test_iter)

        elif strstr(line, 'Test net output'):
          matched = match_evaluation(line)
          detection_eval.append(float(matched.group(1)))
          
  print("TRAIN", train_iteration, train_loss)
  print("TEST", test_iteration, detection_eval)

  # loss
  plt.plot(train_iteration, train_loss, 'k', label='Train loss')
  plt.legend()
  plt.ylabel('Loss')
  plt.xlabel('Number of iterations')
  plt.savefig('loss.png')

  # evaluation
  plt.clf()
  plt.plot(test_iteration, detection_eval, 'r', label='Detection evaluation')
  plt.legend()
  plt.ylabel('Detection_eval')
  plt.xlabel('Number of iterations')
  plt.savefig('evaluation.png')

def match_iteration(line):
  return re.search(r'Iteration (.*),', line)

def match_loss(line):
  return re.search(r'mbox_loss = (.*) \(', line)
  
def match_evaluation(line):
  return re.search(r'detection_eval = (.*)', line)

def process_arguments(argv):
  if len(argv) < 2:
    help()

  log_files = argv[1:]
  return log_files

def help():
  print('Usage: python loss_from_log.py [LOG_FILE]+\n'
        'LOG_FILE is text file containing log produced by caffe.'
        'At least one LOG_FILE has to be specified.'
        'Files has to be given in correct order (the oldest logs as the first ones).'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
