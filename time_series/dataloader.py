import numpy as np

class TrainDataLoader:    
    def __init__(self, file_path, deg, num_vars):
        self.file_path = file_path
        self.deg = deg
        self.num_vars = num_vars
        self.input_x, self.target = self.load_data()
        self.mu, self.sigma = 0, 1
        # self.mu, self.sigma = self.normalise_data()
        self.features = len(self.input_x[0])

    def load_data(self):
        f = open(self.file_path, 'r')
        lines = f.readlines()[1:]
        if self.num_vars == 1:
            target = [[] for i in range(12)]
            for line in lines:
                data = line.strip().split(',')
                x = int(data[0].split('/')[0])
                target[x-1].append(float(data[1]))
            f.close()
            target = np.array([np.array(t).mean() for t in target])
            input_x = []
            for i in range(1,13):
                x = float(i)**np.arange(self.deg+1)
                input_x.append(x)
            input_x = np.array(input_x)
        else:
            input_x, target = [], []
            for line in lines:
                data = line.strip().split(',')
                tokens = data[0].split('/')
                m, yr = float(tokens[0]), float(tokens[2])
                x = []
                for i in range(self.deg + 1):
                    for j in range(self.deg-i+1):
                            x.append((m**i)*(yr**j))
                input_x.append(np.array(x))
                target.append(float(data[1]))
            input_x, target = np.array(input_x), np.array(target)
        return input_x, target
    
    def normalise_data(self):
        mu = self.input_x.mean(0)
        sigma = np.std((self.input_x-mu), axis=0)
        mu[0], sigma[0] = 0.0, 1.0
        self.input_x = (self.input_x-mu)/sigma
        return mu, sigma

    def get_data_queue(self):
        return self.input_x, self.target

class TestDataLoader:
    def __init__(self, file_path, deg, num_vars):
        self.file_path = file_path
        self.deg = deg
        self.num_vars = num_vars
        self.input_x = self.load_data()
        self.features = len(self.input_x[0])

    def load_data(self):
        f = open(self.file_path, 'r')
        lines = f.readlines()[1:]
        input_x = []
        for line in lines:
            data = line.strip().split(',')
            tokens = data[0].split('/')
            m, yr = float(tokens[0]), float(tokens[2])
            if self.num_vars == 1:
                x = m**np.arange(self.deg+1)
            else:
                x = []
                for i in range(self.deg + 1):
                    for j in range(self.deg-i+1):
                            x.append((m**i)*(yr**j))
                x = np.array(x)
            input_x.append(x)
        f.close()
        input_x = np.array(input_x)
        return input_x

    def get_data_queue(self):
        return self.input_x