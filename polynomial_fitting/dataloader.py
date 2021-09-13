import numpy as np

class DataLoader:    
    def __init__(self, file_path, batch_size, poly_deg):
        self.file_path = file_path
        self.batch_size = batch_size
        self.poly_deg = poly_deg
        self.input, self.target = self.load_data()
    
    def load_data(self):
        f = open(self.file_path, 'r')
        target, input_x = [], []
        for line in f.readlines():
            nums = line.strip().split(',')
            target.append(float(nums[0]))
            input_x.append(float(nums[0]))
        input_x, target = np.array(input_x), np.array(target)
        return input_x, target

    def get_data_queue(self):
        data_queue = []
        for i in range(0, len(self.input), self.batch_size):
            x = self.input[i:i+self.batch_size]
            x = x.reshape(-1,1).repeat(self.poly_deg+1, 1)**np.arange(self.poly_deg+1).reshape(1,-1)
            y = self.target[i:i+self.batch_size]
            data_queue.append((x,y))
        return data_queue