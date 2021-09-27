import numpy as np

class LinearModel:
    def __init__(self, in_features, mu, sigma):
        self.weights = np.random.rand(in_features)
        self.mu, self.sigma = mu, sigma

    def pinv_method(self, input_x, target, wt_decay):
        N = len(self.weights)
        theta = np.linalg.inv(np.identity(N)*wt_decay + np.matmul(input_x.T, input_x))
        theta =  np.matmul(theta, input_x.T)
        weights = np.matmul(theta, target)
        self.weights = weights

    def forward(self, input_x, target):
        y = np.matmul(input_x, self.weights)
        loss = ((y-target)**2).mean()/2
        return y, loss

    def get_preds(self, input_x,  mu, sigma):
        input_x = (input_x - mu)/sigma
        y = np.matmul(input_x, self.weights)
        return y

    def train_step(self, input_x, target, lr, wt_decay):
        y, loss = self.forward(input_x, target)
        del_w = ((y-target).reshape(-1,1)*input_x).mean(0) + wt_decay*self.weights
        self.weights = self.weights - lr*del_w
        return loss

    def get_weights(self, mu, sigma):
        weights = self.weights/sigma
        weights[0] -= (weights*mu).sum()
        return weights