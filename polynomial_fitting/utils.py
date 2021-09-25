import os
import numpy as np

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

class Logger():
    def __init__(self, log_path, tag):
        self.log_path = log_path
        self.log_file = os.path.join(log_path, f'{tag}.txt')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.isfile(self.log_file):
            f = open(self.log_file, 'x')
            f.close()
    
    def log(self, logs):
        f = open(self.log_file, 'a')
        f.write(logs)
        f.write('\n')
        print(logs)
        f.close()

def get_output_from_weights(x, weights):
    tokens = weights.split(',')
    weights = np.array([float(wt) for wt in tokens])
    deg = len(weights)
    y = np.matmul(x.reshape(-1,1).repeat(deg, 1)**np.arange(deg).reshape(1,-1), weights)
    return y