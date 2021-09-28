import argparse
import numpy as np
from model import PolynomialFitter
from dataloader import DataLoader
import utils

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="results_1a/", type=str, help = "Directory where logs should be stored")  
    parser.add_argument("--data_file", default="data/gaussian.csv", type=str, help = "Read content from the file")
    parser.add_argument("--split", default=0.8, type=float, help = "Split for train/test set")
    return parser.parse_args()

def train_model(args, epochs, batch_size, deg, lr, lamb, split, tag, 
                log=False, log_wts=False, log_batch=False, log_batch_wts=False):
    model = PolynomialFitter(deg)
    dataloader = DataLoader(args.data_file, batch_size, deg, split)
    train_data_queue = dataloader.get_data_queue()
    test_data_queue = dataloader.get_data_queue(train=False)

    if log or log_wts or log_batch or log_batch_wts:
        logger = utils.Logger(args.log_path, tag)
    for i in range(epochs):
        train_loss, test_loss = utils.AverageMeter(), utils.AverageMeter()
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

if __name__ == '__main__':
    args = setup()

    # For plots of different degree polynomials
    logger = utils.Logger(args.log_path, 'polynomial_deg_wts')
    for d in range(12):
        weights, _, _ = train_using_pinv(args, d, 0, 1)
        weights = ','.join(map(str, weights))
        logger.log(f'Degree {d}: Weights: {weights}')

    # For loss v/s polynomial degree plot
    logger = utils.Logger(args.log_path, 'polynomial_deg_loss')
    for d in range(16):
        _, train_loss, test_loss = train_using_pinv(args, d, 0, 0.2)
        logger.log(f'Degree {d}: Training Loss: {train_loss} Testing Loss: {test_loss}')

    # For loss v/s number of epochs plot
    epochs, batch_size, poly_deg, lr, lamb, split = 100, -1, 10, 2e-3, 0, 0.85
    train_model(args, epochs, batch_size, poly_deg, lr, lamb, split, 'grad_descent_loss', log=True)

    # For plots of different polynomials after some intervals of gradient descent
    epochs, batch_size, poly_deg, lr, lamb, split = 5000, 10, 10, 1e-1, 0, 1
    train_model(args, epochs, batch_size, poly_deg, lr, lamb, split, 'grad_descent_wts', log_wts=True)

    # For loss v/s number of epochs plot for mini batch gradient descent
    epochs, batch_size, poly_deg, lr, lamb, split = 50, 1, 10 , 2e-3, 0, 0.2
    train_model(args, epochs, batch_size, poly_deg, lr, lamb, split, 'stochastic_loss', log_batch=True)

    # For plotting weights in parameter space for Stochastic Gradient Descent
    epochs, poly_deg, lr, lamb, split = 10, 2 , 2e-3, 0, 0.1
    train_model(args, epochs, 1, poly_deg, lr, lamb, split, 'stochastic_wts', log_batch_wts=True)

    # For plots of different polynomials at various values of lambda
    logger = utils.Logger(args.log_path, 'lambda_wts')
    lamb = [0, 1e-2, 1e-1, 1]
    for l in lamb:
        weights, _, _ = train_using_pinv(args, 10, l, 1)
        weights = ','.join(map(str, weights))
        logger.log(f'Lambda {l}: Weights: {weights}')

    # for plots of polynomials using the first 20 points and full dataset
    logger = utils.Logger(args.log_path, 'dataset_wts')
    data_splits = [0.2, 1]
    for d in data_splits:
        weights, _, _ = train_using_pinv(args, 10, 0, d)
        weights = ','.join(map(str, weights))
        logger.log(f'Data split {d}: Weights: {weights}')