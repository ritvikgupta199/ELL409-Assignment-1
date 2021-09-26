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

def train_model(args, batch_size, deg, lr, lamb, split, tag, log=False, log_wts=False):
    model = PolynomialFitter(deg)
    dataloader = DataLoader(args.data_file, batch_size, deg, split)
    train_data_queue = dataloader.get_data_queue()
    test_data_queue = dataloader.get_data_queue(train=False)

    if log or log_wts:
        logger = utils.Logger(args.log_path, tag)
    for i in range(args.epochs):
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
    dataloader = DataLoader(args.data_file, -1, deg, 1)
    train_data_queue = dataloader.get_data_queue()
    x, t = train_data_queue[0] 
    model.pinv_method(x, t, lamb)
    weights = model.get_weights(dataloader.mu, dataloader.sigma)
    return weights

if __name__ == '__main__':
    args = setup()

    #Training the model for various polynomial degrees
    logger = utils.Logger(args.log_path, 'polynomial_deg')
    for d in range(11):
        weights = train_model(args, 100, d, 2e-1, 0, 1, 'polynomial_deg')
        # weights = train_using_pinv(args, d, 0, 1)
        weights = ','.join(map(str, weights))
        logger.log(f'Degree {d}: Weights: {weights}')

    # weights = train_using_pinv(args, 10, 0, 1)
    # weights = ','.join(map(str, weights))
    # logger.log(f'Degree 10: Weights: {weights}')

    # weights = train_model(args, -1, 10, 2e-1, 0, 1, 'polynomial_deg')
    # weights = ','.join(map(str, weights))
    # logger.log(f'Degree 10: Weights: {weights}')

    #Training the model on the tuned hyperparameters, logging weights after each epoch
    batch_size, poly_deg, lr, lamb, split = 10, 4, 2e-3, 0.5, 0.85
    train_model(args, batch_size, poly_deg, lr, lamb, split, 'grad_descent', log=True, log_wts=True)