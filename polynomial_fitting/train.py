import numpy as np
from .dataloader import DataLoader
from .model import PolynomialFitter

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
            for (input_x, target) in train_data_queue:
                loss = model.train_step(input_x, target, args.lr, args.lamb)
                print(f'loss {i}: {loss}')

    else:
        x,t = train_data_queue[0] 
        model.pinv_method(x, t, args.lamb)
        y, loss = model.forward(x, t)
        print(f'Training Loss: {loss}')

        x,t = test_data_queue[0]
        y, loss = model.forward(x, t)
        print(f'Test Loss: {loss}')

    print(f'weights={model.weights}')