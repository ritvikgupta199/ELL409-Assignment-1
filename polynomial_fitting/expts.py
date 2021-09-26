import argparse
import numpy as np
from model import PolynomialFitter
from dataloader import DataLoader
import utils

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="results/", type=str, help = "Directory where logs should be stored")  
    parser.add_argument("--data_file", default="data/gaussian.csv", type=str, help = "Read content from the file")
    parser.add_argument("--epochs", default=5000, type=int, help = "Number of epochs")
    parser.add_argument("--split", default=0.8, type=float, help = "Split for train/test set")
    return parser.parse_args()

def train_model(args, epochs, batch_size, deg, lr, lamb, split, tag, log=False, log_wts=False):
    model = PolynomialFitter(deg)
    dataloader = DataLoader(args.data_file, batch_size, deg, split)
    train_data_queue = dataloader.get_data_queue()
    test_data_queue = dataloader.get_data_queue(train=False)

    if log or log_wts:
        logger = utils.Logger(args.log_path, tag)
    for i in range(epochs):
        train_loss, test_loss = utils.AverageMeter(), utils.AverageMeter()
        for (input_x, target) in train_data_queue:
            loss = model.train_step(input_x, target, lr, lamb)
            train_loss.update(loss, len(input_x))
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

if __name__ == '__main__':
    args = setup()

    #Training the model for various polynomial degrees on entire data
    logger = utils.Logger(args.log_path, 'polynomial_deg')
    for d in range(12):
        weights, _, _ = train_using_pinv(args, d, 0, 1)
        weights = ','.join(map(str, weights))
        logger.log(f'Degree {d}: Weights: {weights}')

    logger = utils.Logger(args.log_path, 'polynomial_deg_loss')
    for d in range(20):
        _, train_loss, test_loss = train_using_pinv(args, d, 0, 0.7)
        logger.log(f'Degree {d}: Training Loss: {train_loss} Testing Loss: {test_loss}')

    #Training the model on the tuned hyperparameters, logging losses after each epoch
    epochs, batch_size, poly_deg, lr, lamb, split = 50, 10, 10, 2e-3, 0, 0.85
    train_model(args, epochs, batch_size, poly_deg, lr, lamb, split, 'grad_descent_loss', log=True)

    #Training the model on the tuned hyperparameters, logging weights after each epoch
    batch_size, poly_deg, lr, lamb, split = 10, 10, 1e-1, 0, 1
    train_model(args, args.epochs, batch_size, poly_deg, lr, lamb, split, 'grad_descent_wts', log_wts=True)

    #Training the model on different values of lambda
    logger = utils.Logger(args.log_path, 'lambda')
    lamb = [0, 1e-2, 1e-1, 1]
    for l in lamb:
        weights, _, _ = train_using_pinv(args, 10, l, 1)
        weights = ','.join(map(str, weights))
        logger.log(f'Lambda {l}: Weights: {weights}')


    
