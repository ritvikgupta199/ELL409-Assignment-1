import argparse
import numpy as np
import utils

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="results_2/", type=str, help = "Directory where logs should be stored")  
    parser.add_argument("--data_file", default="data/train.csv", type=str, help = "Path to data file")
    return parser.parse_args()

if __name__ == '__main__':
    args = setup()

    # For plot of loss v/s degree of curve
    max_deg, lamb = 9, 0.2
    logger = utils.Logger(args.log_path, 'deg_loss')
    for d in range(max_deg+1):
        loss =  utils.train_using_pinv(args, d, lamb)
        logger.log(f'M: {d}: Training Loss: {loss}')