import argparse
import numpy as np
from model import PolynomialFitter
from dataloader import DataLoader
import utils

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=1, type=int, help="Part of code to run")
    parser.add_argument("--method", default="gd", help = "Type of solver")  
    parser.add_argument("--lr", default=0.5, type=float, help = "Learning Rate for gradient descent")
    parser.add_argument("--epochs", default=5, type=int, help = "Number of epochs")
    parser.add_argument("--batch_size", default=10, type=int, help = "Batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "Regularization constant")
    parser.add_argument("--polynomial", default=1, type=int, help = "Degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="data/gaussian.csv", type=str, help = "Read content from the file")
    parser.add_argument("--split", default=0.1, type=float, help = "Split for train/test set")
    return parser.parse_args()

if __name__ == '__main__':
    args = setup()

    model = PolynomialFitter(args.polynomial)
    if args.method == 'pinv':
        args.batch_size = -1
    dataloader = DataLoader(args.X, args.batch_size, args.polynomial, args.split)
    train_data_queue = dataloader.get_data_queue(train=True)
    test_data_queue = dataloader.get_data_queue(train=False)

    if args.method == 'gd':
        for i in range(args.epochs):
            train_loss, test_loss = utils.AverageMeter(), utils.AverageMeter()
            for (input_x, target) in train_data_queue:
                loss = model.train_step(input_x, target, args.lr, args.lamb)
                train_loss.update(loss, len(input_x))
            for (input_x, target) in test_data_queue:
                _, loss = model.forward(input_x, target)
                test_loss.update(loss, len(input_x))
            print(f'Epoch {i+1}: Train Loss {train_loss.avg} | Test Loss {test_loss.avg}')
    else:
        x, t = train_data_queue[0] 
        model.pinv_method(x, t, args.lamb)
        y, loss = model.forward(x, t)
        print(f'Training Loss: {loss}')

        if len(test_data_queue) > 0:
          x, t = test_data_queue[0]
          y, loss = model.forward(x, t)
          print(f'Test Loss: {loss}')

    weights = model.get_weights(dataloader.mu, dataloader.sigma)
    print(f'weights={weights}')