import numpy as np

class TrainDataLoader:    
    def __init__(self, file_path, deg):
        self.file_path = file_path
        self.deg = deg
        self.x_mon, self.target = self.load_data()
        self.mu, self.sigma = self.normalise_data()

    def load_data(self):
        f = open(self.file_path, 'r')
        lines = f.readlines()[1:]
        target = [[] for i in range(12)]
        for line in lines:
            data = line.strip().split(',')
            x = int(data[0].split('/')[0])
            target[x-1].append(float(data[1]))
        f.close()
        target = np.array([np.array(t).mean() for t in target])
        x_mon = []
        for i in range(1,13):
            x = float(i)**np.arange(self.deg+1)
            x_mon.append(x)
        x_mon = np.array(x_mon)
        return x_mon, target
    
    def normalise_data(self):
        mu = self.x_mon.mean(0)
        sigma = np.std((self.x_mon-mu), axis=0)
        mu[0], sigma[0] = 0.0, 1.0
        self.x_mon = (self.x_mon-mu)/sigma
        return mu, sigma

    def get_data_queue(self):
        return self.x_mon, self.target

class TestDataLoader:
    def __init__(self, file_path, deg):
        self.file_path = file_path
        self.deg = deg
        self.x_mon = self.load_data()

    def load_data(self):
        f = open(self.file_path, 'r')
        lines = f.readlines()[1:]
        x_mon = []
        for line in lines:
            data = line.strip().split(',')
            x = float(data[0].split('/')[0])
            x = x**np.arange(self.deg+1)
            x_mon.append(x)
        x_mon = np.array(x_mon)
        f.close()
        return x_mon

    def get_data_queue(self):
        return self.x_mon