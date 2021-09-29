import argparse
import os
import numpy as np
from dataloader import TrainDataLoader, TestDataLoader
from model import LinearModel

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="pinv", help="Type of solver")  
    parser.add_argument("--lr", default=0.01, type=float, help="Learning Rate for gradient descent")
    parser.add_argument("--epochs", default=10000, type=int, help="Number of epochs")
    parser.add_argument("--lamb", default=0.2, type=float, help="Regularization constant")
    parser.add_argument("--degree", default=5, type=int, help="Degree of polynomial to fit")
    parser.add_argument("--num_vars", default=2, type=int, help="Number of in-features to consider")
    parser.add_argument("--results", default="results_2/", type=str, help="Files to store results") 
    parser.add_argument("--train_data", default="data/train.csv", type=str, help="Training data file")
    parser.add_argument("--test_data", default="data/test.csv", type=str, help="Testing data file")
    return parser.parse_args()

if __name__ == '__main__':
    args = setup()
    print(args)

    if args.method == 'pinv':
        args.batch_size = -1
    train_data = TrainDataLoader(args.train_data, args.degree, args.num_vars)
    test_data = TestDataLoader(args.test_data, args.degree, args.num_vars)

    train_data_queue = train_data.get_data_queue()
    test_data_queue = test_data.get_data_queue()

    model = LinearModel(train_data.features, train_data.mu, train_data.sigma)

    if args.method == 'gd':
        for i in range(args.epochs):
            input_x, target = train_data_queue
            loss = model.train_step(input_x, target, args.lr, args.lamb)
            print(f'Epoch {i+1}: Training Loss {loss}')
    else:
        x, t = train_data_queue
        model.pinv_method(x, t, args.lamb)
        y, loss = model.forward(x, t)
        print(f'Training Loss: {loss}')

    x_test = test_data_queue
    y = model.get_preds(x_test, train_data.mu, train_data.sigma)
    print(f'Output: {y}')

    weights = model.get_weights(train_data.mu, train_data.sigma)
    wts_file = os.path.join(args.results, f'weights_{args.num_vars}.txt')
    fw = open(wts_file, 'w')
    fw.write(','.join([str(w) for w in weights]))
    print(f'Weights: {weights}')
    print(f'Weights written in file {wts_file}')

    submit_file = os.path.join(args.results, 'submit.csv')
    fw = open(submit_file, 'w')
    fr = open(args.test_data, 'r')
    fw.write('id,value\n')
    lines = fr.readlines()[1:]
    for (in_x, out_y) in zip(lines, y):
        fw.write(f'{in_x.strip()},{out_y}\n')
    fr.close()
    fw.close()
    print(f'Output written in file {submit_file}')