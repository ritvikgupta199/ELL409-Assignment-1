import argparse
import numpy as np
from model import PolynomialFitter
from dataloader import DataLoader
import utils

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="results_1b/", type=str, help = "Directory where logs should be stored")  
    parser.add_argument("--data_file", default="data/non_gaussian.csv", type=str, help = "Path to data file")
    return parser.parse_args()

if __name__ == '__main__':
    args = setup()

    # For plots of different degree polynomials
    logger = utils.Logger(args.log_path, 'polynomial_deg_wts')
    for d in range(12):
        weights, _, _ = utils.train_using_pinv(args, d, 0, 1)
        weights = ','.join(map(str, weights))
        logger.log(f'Degree {d}: Weights: {weights}')

    # For loss v/s polynomial degree plot
    logger = utils.Logger(args.log_path, 'polynomial_deg_loss')
    for d in range(16):
        _, train_loss, test_loss = utils.train_using_pinv(args, d, 0, 0.2)
        logger.log(f'Degree {d}: Training Loss: {train_loss} Testing Loss: {test_loss}')

    # For finding the values of (y-t)
    logger = utils.Logger(args.log_path, 'noise')
    deg, split, lamb = 11, 1, 0
    model = PolynomialFitter(deg)
    dataloader = DataLoader(args.data_file, -1, deg, split)
    x, t = dataloader.get_data_queue()[0]
    
    model.pinv_method(x, t, lamb)
    y, loss = model.forward(x, t)
    for ns in (y-t):
        logger.log(str(ns))