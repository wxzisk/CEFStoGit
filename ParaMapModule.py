from __future__ import print_function

import sys
import socket
import time
import struct
import json
import copy
import random
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import  BayesianOptimization
from bayes_opt import UtilityFunction
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }

def plot_bo(bo):
    x = np.linspace(0, 1, 1000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    dataframe = pd.DataFrame({'mean':mean,'sigma':sigma})
    dataframe.to_csv("~/prediction.csv",index=False,sep=',')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(bo.space.params.flatten(), -bo.space.target, c="red", s=50, zorder=10, label='Observations')
    plt.plot(x, -mean, '--', color='k', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([- mean + 1.9600 * sigma, -(mean + 1.9600 * sigma)[::-1]]),
              alpha=0.3, label='95% confidence interval')
    plt.legend(loc='upper right', frameon=True, edgecolor='k', fancybox=False, shadow=False)
    plt.ylabel('Normalized Iteration Time', fontdict={'size':12})
    plt.xlabel('Priority Threshold', fontdict={'size':12})
    dataframe = pd.DataFrame({'params':bo.space.params.flatten(), 'target':bo.space.target})
    dataframe.to_csv("~/observation.csv",index=False,sep=',')
    plt.show()

def plot_bo_from_data():
    pred = pd.read_csv("~/prediction.csv")
    obse = pd.read_csv("~/observation.csv")
    mean = pred['mean']
    sigma = pred['sigma']
    params = obse['params']
    target = obse['target']
    x = np.linspace(0, 1, 1000)
    
    plt.figure(figsize=(8.8, 4.3))
    p3 = plt.scatter(params, -target, c="black", s=100, label='Observations')
    p1 = plt.plot(x, -mean, '--', color='k', linewidth=3, label='Prediction')
    p2 = plt.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([- mean + 1.9600 * sigma, -(mean + 1.9600 * sigma)[::-1]]),
              alpha=0.3, color="black", label='95% Confidence Interval')
    plt.legend(loc='upper center', frameon=True, fancybox=False, shadow=False, prop=font1)
    plt.ylabel('Normalized Iteration Time', fontdict={'family':'Times New Roman', 'size':22})
    plt.xlabel('Exploration of Data Priority Assignment', fontdict={'family':'Times New Roman', 'size':22})
    plt.xlim([0,1])
    plt.ylim([0.95,1.2])
    plt.tick_params(labelsize=22)
    x = [0.0, 0.263, 0.5, 0.736, 1.0]
    label_list = ['(0:38)','(10:28)','(19:19)','(28:10)','(38:0)']
    plt.xticks([index for index in x], label_list)
    plt.yticks([0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    #legend1=plt.legend(handles=[p1, p2, p3],loc='upper center', frameon=True, edgecolor='k', fancybox=False, shadow=False)
    #plt.gca().add_artist(legend1)
    plt.show()

def print_point(point, col):
  size = len(point)
  real_p = 0
  for i in range(size):
    real_p = real_p + point['p{}'.format(i)] * (1 - real_p)
    print("p{}: {:.2f}   ".format(i, real_p), end='\t')
    if (i+1) % col == 0:
      real_p = 0
      print()


class LessPrecise(float):
  def __repr__(self):
    return str(self)


def round_to_less(t, n):
  y = copy.deepcopy(t)
  for k, v in y.items():
    v = LessPrecise(round(v, n))
    y[k] = v
  return y


def connect_worke_0(server_ip='127.0.0.1', port=5006):
  try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  except socket.error as msg:
    print('Failed to create socket. Error message : {}'.format(msg))
    sys.exit()

  connected_ = 0
  while (connected_ == 0):
    try:
      sock.connect((server_ip, port))
    except socket.error as msg:
      print('{}. Tring to reconnect.'.format(msg))
      time.sleep(2)
      continue
    connected_ = 1
  print('Socket Connected to ' + server_ip)

  try:
    sock.sendall(bytes('ParameterMapping', encoding="ascii"))
  except socket.error as msg:
    print('Failed to send HELLO message. Error message : {}'.format(msg))
    sys.exit()

  return sock


def encode(s):
  return ' '.join([bin(ord(c)).replace('0b', '') for c in s])


def decode(s):
  return ''.join([chr(i) for i in [int(b, 2) for b in s.split(' ')]])


def recv_meta_info(recv_fd):
  recv_data = recv_fd.recv(12, socket.MSG_WAITALL)
  int_ret_1, int_ret_2, int_ret_3 = struct.unpack('<III', recv_data)
  return int_ret_1, int_ret_2, int_ret_3


def recv_iteration_time(recv_fd):
  recv_data = recv_fd.recv(8, socket.MSG_WAITALL)
  if len(recv_data) != 8:
    print("recv_data length = {}".format(len(recv_data)))
  int64_ret = struct.unpack('<q', recv_data)
  return int64_ret[0]


def send_next_point(send_fd, message, finish):
  message["finish"] = finish
  message_json = json.dumps(message)
  send_fd.sendall(bytes(message_json, encoding="ascii"))


def main():
  parser = argparse.ArgumentParser(description="progrom description")
  parser.add_argument('-s', '--server_ip', type=str, default="127.0.0.1")
  parser.add_argument('-p', '--port', type=int, default=5006)
  parser.add_argument('-m', '--max_iterations', type=int, default=10)
  parser.add_argument('-f', '--plot_figure', action='store_true')
  args = parser.parse_args()

  if args.plot_figure:
    plot_bo_from_data()
    return
  
  server_ip = args.server_ip
  port = args.port
  max_iterations = args.max_iterations

  worker0_fd = connect_worke_0(server_ip, port)
  num_heter_workers, num_priorities, num_parameters = recv_meta_info(worker0_fd)
  col = num_priorities - 1
  num_thresholds = num_heter_workers * col
  parameter_group_size = num_parameters // num_priorities # must guarantee 'exact division'
  print("="*64)
  print("num_heter_workers = {}\n".format(num_heter_workers) + 
        "num_priorities    = {}\n".format(num_priorities) +
        "num_thresholds    = {}\n".format(num_thresholds) +
        "num_parameters    = {}\n".format(num_parameters) +
        "para_group_size   = {}".format(parameter_group_size))

  search_space = {}
  for i in range(num_thresholds):
    search_space['p{}'.format(i)] = (0, 1)

  # https://github.com/fmfn/BayesianOptimization/blob/9b8bafbcad2b85d1ac49ae292756481a4a8e74b6/examples/exploitation_vs_exploration.ipynb
  #utility = UtilityFunction(kind="ucb",kappa=10, xi=0.0)
  #utility = UtilityFunction(kind='ei', kappa=0.0, xi=0.5)
  utility = UtilityFunction(kind='poi', kappa=2.576, xi=0.03) # 0.03 ##0.005
  optimizer = BayesianOptimization(
      f=None,
      pbounds=search_space,
      verbose=0,
      random_state=1,)
  
  # default:
  # [1/m 1/(m-1) 1/(m-2) ...]
  # [1/m 1/(m-1) 1/(m-2) ...]
  # ...
  # [1/m 1/(m-1) 1/(m-2) ...]
  init_point = {}
  for i in range(num_heter_workers):
    for j in range(num_priorities-1):
      if j == 0:
        init_point['p'+str(i*(num_priorities-1))] = parameter_group_size*1.0/num_parameters
      else:
        init_point['p'+str(i*(num_priorities-1)+j)] = \
            parameter_group_size*1.0/num_parameters / (1-j* parameter_group_size*1.0/num_parameters)

  next_point = init_point
  finish = 0
  history = []
  iteration_time = []
  his = round_to_less(next_point, 2)
  #print("new: {}".format(his))
  history.append(his)
  point_cnt = 0
  i = 0
  has_done = 0
  baseline = 1
  while True:
    point_cnt = point_cnt + 1
    print("="*64)
    print('{}:\t'.format(point_cnt))
    print_point(next_point, col)
    #print(his, end="\t")
    iter_time = recv_iteration_time(worker0_fd)
    if baseline == 1:
      baseline = iter_time
    iter_time = iter_time * 1.0 / baseline
    time_start = time.time()
    iteration_time.append(iter_time)
    target = -1 * iter_time
    print("true iter_time = {}".format(iter_time), end="\t")
    optimizer.register(params=next_point,target=target)
    if point_cnt >= max_iterations:
      finish = 1
      last_point = optimizer.max['params']
      his = round_to_less(last_point, 2)
      print()
      print("="*64)
      print("The optimization taget (point) is {}".format(
          -1*optimizer.max['target']))
      print_point(last_point, col)
      print()
      send_next_point(worker0_fd, last_point, finish)
      break
    else:
      next_point = optimizer.suggest(utility)
      his = round_to_less(next_point, 2)
      cnt = 0 # avoid dead lock
      while False:
      #while his in history:
        cnt = cnt + 1
        #print("dup: {}".format(his))
        if (cnt+1) % 5 == 0:
          iter_time = int(iteration_time[history.index(his)] * random.uniform(0.998, 1.002))
          if baseline == 1:
            baseline = iter_time
          iter_time = iter_time * 1.0 / baseline
          target = -1 * iter_time
          point_cnt = point_cnt + 1
          print()
          print("="*64)
          print('{}:\t'.format(point_cnt))
          print_point(next_point, col)
          #print("\n{}".format(his), end="\t")
          print("fake iter_time = {}".format(iter_time), end='\t')
          optimizer.register(params=next_point,target=target)
          if point_cnt == max_iterations:
            finish = 1
            last_point = optimizer.max['params']
            his = round_to_less(last_point, 2)
            print()
            print("="*64)
            print("The optimization taget (point) is {}".format(
                -1*optimizer.max['target']))
            print_point(last_point, col)
            print()
            send_next_point(worker0_fd, last_point, finish)
            has_done = 1
            break
        next_point = optimizer.suggest(utility)
        his = round_to_less(next_point, 2)
        if cnt == 20:
          break # avoid dead lock
      if has_done == 0:
        #print("new: {}".format(his))
        history.append(his)
        send_next_point(worker0_fd, copy.deepcopy(next_point), finish)
    if has_done :
      break
    else:
      time_end = time.time()
      print(time_end - time_start)

  worker0_fd.close()
  if num_thresholds == 1:
    plot_bo(optimizer)


if __name__=="__main__":
  main()
