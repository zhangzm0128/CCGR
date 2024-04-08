import os
import numpy as np
import time
import utils

class DataLoader():
    def __init__(self, config, loader_type, read_mode='memory'):
        '''
        The DataLoader loads data as a batch data
        Inputs: config, mode
            - config: config file for init DataLoader class
            - loader_type: the type of DataLoader, limited in ['train', 'valid', 'predict']
            - read_mode: the mode of DataLoader to load data, limited in ['memory', 'disk']. Default of it is 'memory' 
        Extrnal functions: reset(), get_num_batch(), load_data()
            - reset(): using this func will reset the DataLoader, that is the load pointer will be set as 0
            - get_num_batch(): use this func to get the total number of batches contained in an epoch
            - load_data(): this func loads data as one batch and follows the 'loader_type'. When it is
                ['train', 'valid'] both src_data and label_data will be loaded. When mode is 'predict', only src_data will be
                loaded. Additionally, this func has an extra parameter to control whether to output the filename of the data.
        '''
        
        # get params from DataLoader config
        self.batch_size = config['batch_size']
        
        self.loader_type = loader_type
        if self.loader_type not in ['train', 'valid', 'predict']:
            raise(ValueError, "Error 'loader_type': {}. 'loader_type' is limited in ['train', 'valid', 'predict']")
        else:
            self.dataset_index = config[self.loader_type + '_index']
        
        # get all data name list from the dataset index file
        self.get_data_list()
        
        # set load_data()  
        if read_mode == 'memory':
            self.read_data2memory()
            self.load_data = self.__load_data_memory
        elif read_mode == 'disk':
            self.load_data = self.__load_data_disk
        else:
            raise (ValueError, "Error 'read_mode': {} of DataLoader. 'read_mode' must be in ['memory', 'disk']".format(mode))
        
        # init other params
        self.epoch = -1    
        self.data_pointer = 0
        self.dataset_size = len(self.data_list)
        self.load_index = np.arange(self.dataset_size, dtype=int)
        self.reset()

    def reset(self):
        if self.loader_type in ['train', 'valid']:
            np.random.shuffle(self.load_index)
        self.data_pointer = 0
        self.epoch = self.epoch + 1
        
    def get_num_batch(self):
        return int(np.ceil(self.dataset_size / self.batch_size))
        
    
    def get_data_list(self):
        # train and valid index file format:
        #     input_data_path_0,label_data_path_0
        #     input_data_path_1,label_data_path_1
        #     ...
        # predict index file format
        #     input_data_path_0
        #     input_data_path_1
        #     ...
        self.data_list = []
        self.name_list = []
        if self.loader_type in ['train', 'valid']:
            for x in open(self.dataset_index, 'r'):
                input_data_path, label_data_path = x.replace('\n', '').split(',')
                self.data_list.append([input_data_path, label_data_path])
                self.name_list.append(os.path.basename(input_data_path))
        else:
            for x in open(self.dataset_index, 'r'):
                input_data_path = x.replace('\n', '')
                self.data_list.append([input_data_path])
                self.name_list.append(os.path.basename(input_data_path))
        self.name_list = np.array(self.name_list)
    
    def read_data2memory(self):
        # read all data into the memory when DataLoader is init
        self.input_data_memory = []
        self.label_data_memory = []
        if self.loader_type == 'predict':
            for x in self.data_list:
                input_data_path = x[0]
                self.input_data_memory.append(np.load(input_data_path))
                self.input_data_memory[-1].dtype
        else:
            for x in self.data_list:
                input_data_path, label_data_path = x
                self.input_data_memory.append(np.load(input_data_path))
                self.label_data_memory.append(np.load(label_data_path))  
        
        self.input_data_memory = np.array(self.input_data_memory)
        self.label_data_memory = np.array(self.label_data_memory)
        
    def read_data_from_disk(self, read_index):
        # According to the input params(read_index) to read data
        input_data = []
        label_data = []
        if self.loader_type == 'predict':
            for x in read_index:
                input_data_path = self.data_list[x][0]
                input_data.append(np.load(input_data_path))
            return np.array(input_data)
        else:
            for x in read_index:
                input_data_path = self.data_list[x][0]
                label_data_path = self.data_list[x][1]
                input_data.append(np.load(input_data_path))
                label_data.append(np.load(label_data_path))
            return np.array(input_data), np.array(label_data)
        
                
                
    def __load_data_memory(self, return_names=False):
        # Based on the value of self.data_pointer, generate a new data index to be read
        if self.data_pointer + self.batch_size <= self.dataset_size:
            read_index = self.load_index[np.arange(self.data_pointer, self.data_pointer + self.batch_size)]           
            self.data_pointer = self.data_pointer + self.batch_size
            if self.data_pointer >= self.dataset_size:
                self.reset()
        else:
            read_index = self.load_index[np.arange(self.data_pointer, self.dataset_size)]            
            self.reset()
        
        # Generate name list of the loading data    
        if return_names == True:
            # batch_names = self.name_list[self.load_index[self.data_pointer: self.data_pointer + self.batch_size]]
            batch_names = self.name_list[read_index]
            
        # Get batch data and return them    
        if self.loader_type != 'predict':
            batch_input_data = self.input_data_memory[read_index]
            batch_label_data = self.label_data_memory[read_index]
            if return_names == True:
                return batch_input_data, batch_label_data, batch_names
            else:
                return batch_input_data, batch_label_data
        else:
            batch_input_data = self.input_data_memory[read_index]
            if return_names == True:
                return batch_input_data, batch_names
            else:
                return batch_input_data
            
    def __load_data_disk(self, return_names=False):
        # Based on the value of self.data_pointer, generate a new data index to be read
        if self.data_pointer + self.batch_size <= self.dataset_size:
            read_index = self.load_index[np.arange(self.data_pointer, self.data_pointer + self.batch_size)]
            self.data_pointer = self.data_pointer + self.batch_size
            if self.data_pointer >= self.dataset_size:
                self.reset()
        else:
            read_index = self.load_index[np.arange(self.data_pointer, self.dataset_size)]
            self.reset()
        
        # Generate name list of the loading data     
        if return_names == True:
            batch_names = self.name_list[read_index]
        
        # Get batch data and return them    
        if self.loader_type != 'predict':
            batch_input_data, batch_label_data = self.read_data_from_disk(read_index)
            if return_names == True:
                return batch_input_data, batch_label_data, batch_names
            else:
                return batch_input_data, batch_label_data
        else:
            batch_input_data = self.read_data_from_disk(read_index)
            if return_names == True:
                return batch_input_data, batch_names
            else:
                return batch_input_data
            
'''        
# Here is the test for DataLoader

config = {'batch_size': 16, 
          'train_index': '/home/deeplearning/UltrasoundData/train_index.txt', 
          'valid_index': '/home/deeplearning/UltrasoundData/valid_index.txt',
          'predict_index': '/home/deeplearning/UltrasoundData/valid_index.txt'}
    
train_data_loader = DataLoader(config, 'train', 'memory')
# predict_data_loader = DataLoader(config, 'predict', 'memory')

start_time = time.time()
for i in range(10 * train_data_loader.get_num_batch()):
    x, y = train_data_loader.load_data()
    print(x.shape, y.shape, x.dtype, y.dtype)
print(utils.format_runtime(time.time() - start_time))
    
# for i in range(predict_data_loader.get_num_batch()):
#     x, names = predict_data_loader.load_data(return_names=True)
#     print(x.shape, names, predict_data_loader.epoch)
'''
