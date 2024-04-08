import os
import time
import numpy as np
from utils.utils import format_runtime

import torch 
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F

from network.loss import Loss 

class Train:
    def __init__(self, config, logger, net, train_data_loader, val_data_loader, device):
        self.config = config
        
        self.logger = logger
        self.net = net
        
        for param_tensor in self.net.state_dict():
            print(param_tensor, "\t", self.net.state_dict()[param_tensor].size()) 
        
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        
        self.device = device
                
        self.loss = Loss(config['train_params']['loss'], self.device).loss()
        self.lr = config['train_params']['learning_rate']
        self.opt = config['train_params']['optimizer']
        
        self.epoch = config['train_params']['epoch']
        self.show_steps = config['train_params']['show_steps']
        self.save_mode = config['train_params']['save_mode']
        
        self.stop_epoch = config['train_params']['early_stop_epoch']
        self.adjust_epoch = config['train_params']['adjust_lr_epoch']
        self.lr_decay = config['train_params']['lr_decay']
        self.no_improve = 0
        self.stopper = False
        self.best_val_loss = None
        self.set_opt()
        
    def set_opt(self):
        if 'opt_args' not in self.config:
            self.opt = eval('optim.' + self.opt)(self.net.parameters(), lr=self.lr)
        else:
            self.opt = eval('optim.' + self.opt)(self.net.parameters(), self.lr, self.config['train_params']['opt_args'])
        
        
            
    def early_stop(self):
        '''
        Set early stop strategy for training
        '''
        if self.best_val_loss is None:
            self.best_val_loss = np.mean(self.val_loss)
        else:
            if np.mean(self.val_loss) < self.best_val_loss:
                self.no_improve = 0
                self.best_val_loss = np.mean(self.val_loss)
            else:
                self.no_improve += 1
        if self.no_improve == self.stop_epoch:
            self.stopper = True
            
    def adjust_lr(self):
        if self.best_val_loss is None:
            self.best_val_loss = np.mean(self.val_loss)
        else:
            if np.mean(self.val_loss) < self.best_val_loss:
                self.no_improve = 0
                self.best_val_loss = np.mean(self.val_loss)
            else:
                self.no_improve += 1
        if self.no_improve == self.adjust_epoch:
            for x in self.opt.param_groups:
                x['lr'] = x['lr'] * self.lr_decay
            
            self.no_impove = 0


    def train(self):
        '''
        Training process
        '''
        epoch_num_batch = self.train_data_loader.get_num_batch()

        train_loss = []

        step_time = time.time()
        # train start
        # current_epoch = 0
        # while not self.stopper:
        for current_epoch in range(self.epoch):
            for train_iter in range(epoch_num_batch):
                x_batch, y_batch = self.train_data_loader.load_data()
                # x_batch = Variable(torch.FloatTensor(x_batch).to(self.device), requires_grad=False)
                # y_batch = Variable(torch.FloatTensor(y_batch).to(self.device), requires_grad=False)
                
                x_batch = torch.complex(torch.FloatTensor(x_batch.real).to(self.device), torch.FloatTensor(x_batch.imag).to(self.device)).to(self.device)
                y_batch = torch.FloatTensor(y_batch).to(self.device)
                
                self.opt.zero_grad()
                output = self.net(x_batch)
                output = output.squeeze(1)
                
                loss = self.loss(output, y_batch)
                loss.backward()
                train_loss.append(loss.item())
                
                self.opt.step()
                
                if (train_iter + current_epoch * epoch_num_batch) % self.show_steps == 0:
                    print('Train Epoch: {} -- Iter: {} -- Loss: {:.4} -- Time: {}'.format(current_epoch, 
                                                                                          train_iter, np.mean(train_loss), 
                                                                                          format_runtime(time.time() - step_time)))
                    self.logger.write_train_log(current_epoch, train_iter, np.mean(train_loss), time.time() - step_time)
                    step_time = time.time()
                    train_loss = []
                    
            self.val_loss, val_time = self.valid(current_epoch)
            self.logger.save_model(self.net, self.val_loss, mode=self.save_mode)
            self.logger.write_valid_log(current_epoch, 0, self.val_loss, val_time)
            
            # self.early_stop()
            # if self.stopper:
            #     break
            # current_epoch = current_epoch + 1
            step_time = time.time()
            self.adjust_lr()

    def valid(self, current_epoch):
        self.val_data_loader.reset()
        val_loss = []
        self.net.eval()
        step_time = time.time()
        for val_iter in range(self.val_data_loader.get_num_batch()):
            x_batch, y_batch = self.val_data_loader.load_data()
            # x_batch = Variable(torch.FloatTensor(x_batch).to(self.device), requires_grad=False)
            # y_batch = Variable(torch.FloatTensor(y_batch).to(self.device), requires_grad=False)
            x_batch = torch.complex(torch.FloatTensor(x_batch.real).to(self.device), torch.FloatTensor(x_batch.imag).to(self.device)).to(self.device)
            y_batch = torch.FloatTensor(y_batch).to(self.device)
            
            
            output = self.net(x_batch)
            output = output.squeeze(1)
            
            loss = self.loss(output, y_batch)
            val_loss.append(loss.item())
            
            x_batch, y_batch = None, None
            output, loss = None, None
        
        val_time = time.time() - step_time
        print('Valid Epoch: {} ------------- Loss: {:.4} -- Time: {}'.format(current_epoch, np.mean(val_loss),
                                                                             format_runtime(val_time)))
        self.net.train()
        return np.mean(val_loss), val_time


   
            
