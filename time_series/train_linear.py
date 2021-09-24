import numpy as np
from .dataloader import DataLoader
from .model import LinearModel

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

def train_linear(args):
  in_features = 4
  if args.method == 'gd':
      train_data = DataLoader(args.train, args.batch_size, train=True)
      test_data = DataLoader(args.test, args.batch_size, train=False)
  else:
      train_data = DataLoader(args.train, -1, train=True)
      test_data = DataLoader(args.test, -1, train=False)
  train_data_queue = train_data.get_data_queue()
  test_data_queue = test_data.get_data_queue()

  model = LinearModel(in_features, train_data.mu, train_data.sigma)

  if args.method == 'gd':
    for i in range(args.epochs):
      train_loss = AverageMeter()
      for (input_x, target) in train_data_queue:
        loss = model.train_step(input_x, target, args.lr, args.lamb)
        train_loss.update(loss, len(input_x))
      print(f'Epoch {i+1}: Train Loss {train_loss.avg}')
  else:
    x, t = train_data_queue[0]
    model.pinv_method(x, t, args.lamb)
    y, loss = model.forward(x, t)
    print(f'Training Loss: {loss}')

  y = []
  for (input_x, _) in test_data_queue:
    y.append(model.get_preds(input_x))
  print(f'Output: {y}')