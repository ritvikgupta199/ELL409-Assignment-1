import os
import numpy as np
from dataloader import TrainDataLoader
from model import LinearModel

class Logger():
    def __init__(self, log_path, tag):
        self.log_path = log_path
        self.log_file = os.path.join(log_path, f'{tag}.txt')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
    
    def log(self, logs):
        f = open(self.log_file, 'a')
        f.write(logs)
        f.write('\n')
        print(logs)
        f.close()

def train_using_pinv(args, deg, lamb):
    train_data = TrainDataLoader(args.data_file, deg, 2)
    x, t = train_data.get_data_queue()
    model = LinearModel(train_data.features, train_data.mu, train_data.sigma)
    model.pinv_method(x, t, lamb)
    _, loss = model.forward(x, t)
    return loss

def get_output_from_weights(x, weights):
    tokens = weights.split(',')
    weights = np.array([float(wt) for wt in tokens])
    deg = len(weights)
    y = np.matmul(x.reshape(-1,1).repeat(deg, 1)**np.arange(deg).reshape(1,-1), weights)
    return y