import numpy as np

class DataLoader:    
    def __init__(self, file_path, batch_size, poly_deg, split):
        self.file_path = file_path
        self.poly_deg = poly_deg
        self.train_input, self.train_target, self.test_input, self.test_target = self.load_data(split)
        self.mu, self.sigma = self.normalise_data()        
        self.batch_size = len(self.train_input) if batch_size == -1 else batch_size

    def load_data(self, split):
        f = open(self.file_path, 'r')
        input_x, target = [], []
        lines = f.readlines()
        split = int(split*len(lines))
        for line in lines:
            nums = line.strip().split(',')
            x = float(nums[0])**np.arange(self.poly_deg+1)
            input_x.append(x)
            target.append(float(nums[1]))
        f.close()
        train_input, train_target = np.array(input_x[:split]), np.array(target[:split])
        test_input, test_target = np.array(input_x[split:]), np.array(target[split:])
        return train_input, train_target, test_input, test_target
    
    def normalise_data(self):
        mu = self.train_input.mean(0)
        sigma = np.std((self.train_input-mu), axis=0)
        mu[0], sigma[0] = 0.0, 1.0
        self.train_input = (self.train_input-mu)/sigma
        if len(self.test_input) > 0:
            self.test_input = (self.test_input-mu)/sigma
        return mu, sigma

    def get_data_queue(self, train = True):
        data_queue = []
        if train:
            for i in range(0, len(self.train_input), self.batch_size):
                x = self.train_input[i:i+self.batch_size]
                y = self.train_target[i:i+self.batch_size]
                data_queue.append((x,y))
        else:
            for i in range(0, len(self.test_input), self.batch_size):
                x = self.test_input[i:i+self.batch_size]
                y = self.test_target[i:i+self.batch_size]
                data_queue.append((x,y))
        return data_queue