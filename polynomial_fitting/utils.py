import os
import numpy as np
from model import PolynomialFitter
from dataloader import DataLoader

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
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
    
    def log(self, logs):
        f = open(self.log_file, 'a')
        f.write(logs)
        f.write('\n')
        print(logs)
        f.close()

def train_model(args, epochs, batch_size, deg, lr, lamb, split, tag, 
                log=False, log_wts=False, log_batch=False, log_batch_wts=False):
    model = PolynomialFitter(deg)
    dataloader = DataLoader(args.data_file, batch_size, deg, split)
    train_data_queue = dataloader.get_data_queue()
    test_data_queue = dataloader.get_data_queue(train=False)

    if log or log_wts or log_batch or log_batch_wts:
        logger = Logger(args.log_path, tag)
    for i in range(epochs):
        train_loss, test_loss = AverageMeter(), AverageMeter()
        for (j, (input_x, target)) in enumerate(train_data_queue):
            loss = model.train_step(input_x, target, lr, lamb)
            train_loss.update(loss, len(input_x))
            if log_batch:
                logger.log(f'Epoch {i+1}: Batch {j+1}: Training Loss: {train_loss.avg}')
            if log_batch_wts:
                wts = ','.join(map(str, model.get_weights(dataloader.mu, dataloader.sigma)))
                logger.log(f'Epoch {i+1}: Batch {j+1} Weights: {wts}')
        for (input_x, target) in test_data_queue:
            _, loss = model.forward(input_x, target)
            test_loss.update(loss, len(input_x))
        if log:
            logger.log(f'Epoch {i+1}: Train Loss {train_loss.avg} | Test Loss {test_loss.avg}')
        if log_wts:
            wts = ','.join(map(str, model.get_weights(dataloader.mu, dataloader.sigma)))
            logger.log(f'Epoch {i+1}: Weights: {wts}')

    weights = model.get_weights(dataloader.mu, dataloader.sigma)
    return weights

def train_using_pinv(args, deg, lamb, split):
    model = PolynomialFitter(deg)
    dataloader = DataLoader(args.data_file, -1, deg, split)
    x, t = dataloader.get_data_queue()[0]
    model.pinv_method(x, t, lamb)
    y, train_loss = model.forward(x, t)
    test_loss = 0
    if split < 1:
        x, t = dataloader.get_data_queue(train=False)[0]
        y, test_loss = model.forward(x,t)

    weights = model.get_weights(dataloader.mu, dataloader.sigma)
    return weights, train_loss, test_loss

def get_output_from_weights(x, weights):
    tokens = weights.split(',')
    weights = np.array([float(wt) for wt in tokens])
    deg = len(weights)
    y = np.matmul(x.reshape(-1,1).repeat(deg, 1)**np.arange(deg).reshape(1,-1), weights)
    return y

def get_weights_from_text(weights):
    tokens = weights.split(',')
    weights = np.array([float(wt) for wt in tokens])
    return weights