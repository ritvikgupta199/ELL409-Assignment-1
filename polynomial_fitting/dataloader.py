import numpy as np

class DataLoader:    
    def __init__(self, file_path, batch_size, poly_deg, split):
        self.file_path = file_path
        self.poly_deg = poly_deg
        self.train_input, self.train_target, self.test_input, self.test_target = self.load_data(split)
        self.batch_size = len(self.train_input) if batch_size == -1 else batch_size

    def load_data(self, split):
        f = open(self.file_path, 'r')
        target, input_x = [], []
        lines = f.readlines()
        split = int(split*len(lines))
        for line in lines:
            nums = line.strip().split(',')
            target.append(float(nums[0]))
            input_x.append(float(nums[0]))
        train_input, train_target = np.array(input_x[:split]), np.array(target[:split])
        test_input, test_target = np.array(input_x[split:]), np.array(target[split:])
        return train_input, train_target, test_input, test_target

    def get_data_queue(self, train = True):
        data_queue = []
        if train:
            for i in range(0, len(self.train_input), self.batch_size):
                x = self.train_input[i:i+self.batch_size]
                x = x.reshape(-1,1).repeat(self.poly_deg+1, 1)**np.arange(self.poly_deg+1).reshape(1,-1)
                y = self.train_target[i:i+self.batch_size]
                data_queue.append((x,y))
        else:
            for i in range(0, len(self.train_input), self.batch_size):
                x = self.test_input[i:i+self.batch_size]
                x = x.reshape(-1,1).repeat(self.poly_deg+1, 1)**np.arange(self.poly_deg+1).reshape(1,-1)
                y = self.test_target[i:i+self.batch_size]
                data_queue.append((x,y))
        return data_queue