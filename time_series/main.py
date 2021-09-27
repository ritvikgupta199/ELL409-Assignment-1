import argparse
import numpy as np
from dataloader import TrainDataLoader
from model import LinearModel

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="pinv", help = "Type of solver")  
    parser.add_argument("--lr", default=0.01, type=float, help = "Learning Rate for gradient descent")
    parser.add_argument("--epochs", default=100, type=int, help = "Number of epochs")
    parser.add_argument("--lamb", default=0, type=float, help = "Regularization constant")
    parser.add_argument("--result", default="results_2/submit.csv", type=str, help = "Files to store plots")  
    parser.add_argument("--train_data", default="data/train.csv", type=str, help = "Training data file")
    parser.add_argument("--test_data", default="data/test.csv", type=str, help = "Testing data file")
    return parser.parse_args()

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


if __name__ == '__main__':
    args = setup()
    print(args)
    in_features = 4

    if args.method == 'pinv':
        args.batch_size = -1
    train_data = TrainDataLoader(args.train_data)
    # test_data = DataLoader(args.test_data, args.batch_size)

    train_data_queue = train_data.get_data_queue()
    # test_data_queue = test_data.get_data_queue()

    model = LinearModel(in_features, train_data.mu, train_data.sigma)

    if args.method == 'gd':
        for i in range(args.epochs):
            input_x, target = train_data_queue
            loss = model.train_step(input_x, target, args.lr, args.lamb)
            print(f'Epoch {i+1}: Train Loss {loss}')
    else:
        x, t = train_data_queue
        model.pinv_method(x, t, args.lamb)
        y, loss = model.forward(x, t)
        print(f'Training Loss: {loss}')

    # y = []
    # for (input_x, _) in test_data_queue:
    #     y.append(model.get_preds(input_x, train_data.mu, train_data.sigma))
    # y = np.concatenate(y, axis=0)
    # print(f'Output: {y}')

    weights = model.get_weights(train_data.mu, train_data.sigma)
    print(f'Weights: {weights}')

    # fw = open(args.result, 'w')
    # fr = open(args.test_data, 'r')
    # fw.write('id,value\n')
    # lines = fr.readlines()[1:]
    # for (in_x, out_y) in zip(lines, y):
    #     fw.write(f'{in_x.strip()},{out_y}\n')
    # fr.close()
    # fw.close()
    # print(f'Output written in file {args.result}')
