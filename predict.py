import os
import time

import numpy as np
import torch

from utils.utils import format_runtime


class Predict:
    def __init__(self, config, logger, net, pred_data_loader, device):
        self.config = config
        self.net = net.to(device)
        self.pred_data_loader = pred_data_loader
        self.device = device
        self.logger = logger
        
        self.logger.set_predict_save()
        self.net.eval()
        
    def predict(self):
        step_time = time.time()
        for pred_iter in range(self.pred_data_loader.get_num_batch()):
            x_batch, names = self.pred_data_loader.load_data(return_names=True)
            x_batch = torch.complex(torch.FloatTensor(x_batch.real).to(self.device), torch.FloatTensor(x_batch.imag).to(self.device)).to(self.device)
            
            
            time_0 = time.time()
            output = self.net(x_batch)
            print('Pred time: ' + format_runtime(time.time() - time_0))
            
            
            output = output.squeeze(1).cpu().detach().numpy()
            print('Prediction Completion: {} -- Time: {}'.format(', '.join(names), format_runtime(time.time() - step_time)))
            
            self.logger.write_predict(output, names)
            self.logger.write_predict_image(output, names)
            step_time = time.time()
            
