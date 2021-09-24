import numpy as np
from .dataloader import DataLoader
from .model import PolynomialFitter

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

def fit_polynomial(args):
    model = PolynomialFitter(args.polynomial)

    if args.method == 'gd':
        dataloader = DataLoader(args.X, args.batch_size, args.polynomial, args.split)
    else:
        dataloader = DataLoader(args.X, -1, args.polynomial, args.split)
    train_data_queue = dataloader.get_data_queue(train=True)
    test_data_queue = dataloader.get_data_queue(train=False)

    if args.method == 'gd':
        for i in range(args.epochs):
            train_loss, test_loss = AverageMeter(), AverageMeter()
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