import argparse
import numpy as np
import utils

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="results_1a/", type=str, help = "Directory where logs should be stored")  
    parser.add_argument("--data_file", default="data/gaussian.csv", type=str, help = "Path to data file")
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

    # For loss v/s number of epochs plot
    epochs, batch_size, poly_deg, lr, lamb, split = 100, -1, 10, 2e-3, 0, 0.85
    utils.train_model(args, epochs, batch_size, poly_deg, lr, lamb, split, 'grad_descent_loss', log=True)

    # For plots of different polynomials after some intervals of gradient descent
    epochs, batch_size, poly_deg, lr, lamb, split = 5000, 10, 10, 1e-1, 0, 1
    utils.train_model(args, epochs, batch_size, poly_deg, lr, lamb, split, 'grad_descent_wts', log_wts=True)

    # For loss v/s number of epochs plot for Stochastic gradient descent
    epochs, batch_size, poly_deg, lr, lamb, split = 50, 1, 10 , 2e-3, 0, 0.2
    utils.train_model(args, epochs, batch_size, poly_deg, lr, lamb, split, 'stochastic_loss', log_batch=True)

    # For plotting weights in parameter space for Stochastic gradient descent
    epochs, poly_deg, lr, lamb, split = 10, 2 , 2e-3, 0, 0.1
    utils.train_model(args, epochs, 1, poly_deg, lr, lamb, split, 'stochastic_wts', log_batch_wts=True)

    # For plots of different polynomials at various values of lambda
    logger = utils.Logger(args.log_path, 'lambda_wts')
    lamb = [0, 1e-2, 1e-1, 1]
    for l in lamb:
        weights, _, _ = utils.train_using_pinv(args, 10, l, 1)
        weights = ','.join(map(str, weights))
        logger.log(f'Lambda {l}: Weights: {weights}')

    # For plots of polynomials using the first 20 points and full dataset
    logger = utils.Logger(args.log_path, 'dataset_wts')
    data_splits = [0.2, 1]
    for d in data_splits:
        weights, _, _ = utils.train_using_pinv(args, 10, 0, d)
        weights = ','.join(map(str, weights))
        logger.log(f'Data split {d}: Weights: {weights}')