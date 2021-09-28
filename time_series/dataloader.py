import numpy as np

class TrainDataLoader:    
    def __init__(self, file_path):
        self.file_path = file_path
        self.input_x, self.target = self.load_data()
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
        input_x = []
        for i in range(1,13):
            x = float(i)**np.arange(4)
            input_x.append(x)
        input_x = np.array(input_x)
        return input_x, target
    
    def normalise_data(self):
        mu = self.input_x.mean(0)
        sigma = np.std((self.input_x-mu), axis=0)
        mu[0], sigma[0] = 0.0, 1.0
        self.input_x = (self.input_x-mu)/sigma
        return mu, sigma

    def get_data_queue(self):
        return self.input_x, self.target