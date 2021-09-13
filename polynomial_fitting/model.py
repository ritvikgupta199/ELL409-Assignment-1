import numpy as np

class PolynomialFitter:
    def __init__(self, poly_deg):
        self.poly_deg = poly_deg
        self.weights = np.random.rand(poly_deg+1)

    def forward(self, input_x, target):
        y = np.matmul(input_x, weights)
        loss = ((y-target)**2).mean()
        return y, loss

    def train_step(self, input_x, target, lr):
        y, loss = self.forward(input_x, target)
        del_w = input_x.mean(0)
        self.weights = self.weights - lr*del_w
        return loss

    