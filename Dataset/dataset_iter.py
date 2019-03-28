import numpy as np
from sklearn.preprocessing import normalize
import pdb
from utils import *
from termcolor import colored

class dataset_iterator:
    def __init__(self,data_file_path,label_file_path,
        batch_size = 32,n_points =1000):
        self.data  = np.load(data_file_path)
        print('Training Dataset is loaded from ', colored(data_file_path, 'green'))
        self.label = np.load(label_file_path)
        print('label is loaded from ',colored( label_file_path, 'green'))
        self.batch_size = batch_size
        self.n_points = n_points
        self.current_sample = 0
        self.iter_num = 0

    def preprocess_data(self,data_sample):
        data_sample = np.array(np.where(data_sample == 1)).T
        return normalize(data_sample)

    def get_batch(self):
        if(self.batch_size + self.current_sample > len(self.data)):
            self.iter_num += 1
            self.current_sample  = 0
            self.data = np.random.shuffle(self.data)
        batch_pre_data = self.data[self.current_sample: self.current_sample + self.batch_size]
        batch_label = self.label[self.current_sample: self.current_sample + self.batch_size]
        batch_data = np.zeros((self.batch_size,self.n_points,3))
        for i in range(self.batch_size):
            batch_data[i] = sample_points(self.preprocess_data(batch_pre_data[i][0]), self.n_points)
        # pdb.set_trace()
        batch_data = np.array(batch_data).reshape(self.batch_size, self.n_points,3)
        self.current_sample += self.batch_size
        return batch_data, batch_label, self.current_sample, self.iter_num



if __name__ == '__main__':
    sample  = dataset_iterator('features.npy','targets.npy')
    data, label, b_1 , iter_n = sample.get_batch()
    pdb.set_trace()
    print('Done')
