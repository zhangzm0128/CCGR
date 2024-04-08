import os
import time
import numpy as np
from utils.utils import format_runtime

import torch 
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F

from network.loss import Loss 

class Test:
    def __init__(self, config, logger, net, val_data_loader, device):
        self.config = config
        self.net = net
        self.val_data_loader = val_data_loader
        self.device = device
        self.logger = logger
        
        self.loss = Loss(config['train_params']['loss'], self.device).loss()
        
        self.net.eval()
    
    def test(self):
        test_loss = []
        test_time = []
        test_epoch = []
        for x in os.listdir(self.logger.weight_save_path):
            model_name = x.replace('.pkl', '')
            if 'best' in model_name:
                continue
            
            self.net = self.logger.load_model(device=self.device, model_name=model_name)
            val_loss, val_time = self.valid(model_name)
            test_loss.append(val_loss)
            test_time.append(val_time)
            test_epoch.append(int(model_name.replace('model_', '')))
        test_file = open('test.csv', 'w')
        test_file.write('Epoch,Loss,Time\n')
        write_line = '{},{},{}\n'
        for x in np.argsort(test_epoch):
            test_file.write(write_line.format(test_epoch[x], test_loss[x], format_runtime(test_time[x])))
            
    
        
    def valid(self, model_name):
        self.val_data_loader.reset()
        val_loss = []
        self.net.eval()
        step_time = time.time()
        for val_iter in range(self.val_data_loader.get_num_batch()):
            x_batch, y_batch = self.val_data_loader.load_data()
            x_batch = torch.complex(torch.FloatTensor(x_batch.real).to(self.device), torch.FloatTensor(x_batch.imag).to(self.device)).to(self.device)
            y_batch = torch.FloatTensor(y_batch).to(self.device)
            
            
            output = self.net(x_batch)
            output = output.squeeze(1)
            
            loss = self.loss(output, y_batch)
            val_loss.append(loss.item())

        
        val_time = time.time() - step_time
        print('Valid Model: {} ------------- Loss: {:.4} -- Time: {}'.format(model_name, np.mean(val_loss),
                                                                             format_runtime(val_time)))
        return np.mean(val_loss), val_time
            
