import numpy as np

class DataLoader:    
    def __init__(self, file_path, batch_size, train=True):
        self.file_path = file_path
        self.train = train
        self.input_x, self.target = self.load_data()
        self.mu, self.sigma = self.normalise_data()    
        self.batch_size = len(self.input_x) if batch_size == -1 else batch_size

    def load_data(self):
        f = open(self.file_path, 'r')
        lines = f.readlines()[1:]
        input_x, target = [], []
        for line in lines:
            data = line.strip().split(',')
            x = [float(n) for n in data[0].split('/')]
            input_x.append(x)
            if self.train:
                target.append(float(data[1]))
        f.close()
        input_x, target = np.array(input_x), np.array(target)
        return input_x, target
    
    def normalise_data(self):
        mu = self.input_x.mean(0)
        sigma = np.sqrt(np.square((self.input_x-mu)).mean(0))
        mu[1], sigma[1] = 0.0, 1.0
        self.input_x = (self.input_x-mu)/sigma
        if self.train:
            mu_t = self.target.mean(0)
            sigma_t = np.sqrt(np.square((self.target-mu_t)).mean(0))
            self.target = (self.target-mu_t)/sigma_t
            return mu_t, sigma_t
        return 0, 0

    def get_data_queue(self):
        data_queue = []
        for i in range(0, len(self.input_x), self.batch_size):
            x = self.input_x[i:i+self.batch_size]
            y = []
            if self.train:
                y = self.target[i:i+self.batch_size]
            data_queue.append((x,y))
        return data_queue