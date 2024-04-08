import os
import numpy as np
import torch
import torch.nn as nn
# from utils.utils import real2complex
from torch.autograd import Variable
import torch.nn.functional as F
import network.complex_func as CF
from torch.nn import init

EPSILON = 1e-6
POSINF = 2e32
NEGINF = -2e32


class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, device='cuda'):
        super(ComplexConv, self).__init__()
        # self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias).to(device)
        # self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias).to(device)
        
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)# .to(device)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)# .to(device)
        
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = device
    
    def init_weight(self):
        real_conv_weight, imag_conv_weight = CF.init_complex_conv_weight(self.kernel_size, self.in_channels, self.out_channels, device=self.device)
        # print(self.real_conv.weight.data)
        # print(real_conv_weight)
        # print('-*-*-')
        

        self.real_conv.weight.data = nn.Parameter(real_conv_weight, requires_grad=True)
        self.imag_conv.weight.data = nn.Parameter(imag_conv_weight, requires_grad=True)
        
            
        # nn.init.constant_(self.real_conv.weight.data, real_conv_weight)
        # nn.init.constant_(self.imag_conv.weight.data, imag_conv_weight)
    
    def forward(self, x):
        # print(x.shape)
        return CF.real2complex(self.real_conv, self.imag_conv, x)


class Complex2RealConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, device='cuda'):
        super(Complex2RealConv, self).__init__()
        self.complex2real_conv = nn.Conv2d(in_channels*2, out_channels, kernel_size, stride, padding, bias=bias)# .to(device)
        # self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    
    
    def forward(self, x):
        concat_input = torch.cat([x.real, x.imag], dim=1)
        return self.complex2real_conv(concat_input)
            
            
class ComplexDropoutRealFormat(nn.Module):
    def __init__(self, prob=0.5):
        super(ComplexDropoutRealFormat, self).__init__()
        self.dropout = nn.Dropout(prob)
    def forward(self, x):
        mask = torch.ones_like(x, dtype=torch.float32)
        mask = self.dropout(mask)
        return x * mask
        

class ComplexConvGRUCell_RealGate(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, padding=0, bias=True, dropout=None, device='cpu'):
        super(ComplexConvGRUCell_RealGate, self).__init__()
        self.device = device
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.reset_gate  = Complex2RealConv(input_size+hidden_size, hidden_size, kernel_size, stride, padding, bias, device=device)
        self.update_gate = Complex2RealConv(input_size+hidden_size, hidden_size, kernel_size, stride, padding, bias, device=device)
        self.output_gate = ComplexConv(input_size+hidden_size, hidden_size, kernel_size, stride, padding, bias, device=device)
        self.dropout = dropout
                
        # self.output_gate_activation = CF.modRelu(self.device)
        
        # init weight
        # self.reset_gate.init_weight()
        # self.update_gate.init_weight()
        self.output_gate.init_weight()
        # init weight
        

    
    def forward(self, input_data, state_data=None):
        if len(input_data.size()) == 3:
            input_data = torch.unsqueeze(input_data, 1)
        batch_size = input_data.size()[0]
        if state_data is None:
            state_size = [batch_size, self.hidden_size] + list(input_data.size()[2:])
            state_data = Variable(torch.zeros(state_size).type(torch.complex64)).to(self.device)
        concat_input = torch.cat([input_data, state_data], dim=1)
        
        update = torch.sigmoid(self.update_gate(concat_input))
        reset  = torch.sigmoid(self.reset_gate(concat_input))
        
        concat_reset = torch.cat([input_data, state_data * reset], dim=1) # Here, 'state_data * reset' is waiting for discussion. Mayby the other mul format is better.
        if self.dropout is not None:
            concat_reset = self.dropout(concat_reset)
        output = CF.complex_split_tanh(self.output_gate(concat_reset))

        next_state = state_data * update + output * (1 - update) # Here, it has the same problem as the above mentioned
        
        # print('next_state:', torch.any(torch.isnan(next_state)), torch.sum(torch.isnan(next_state)))
        
        return next_state


        
        
class ComplexConvGRU(nn.Module):
    def __init__(self, config, device='cpu'):
        super(ComplexConvGRU, self).__init__()
        self.device = device
        
        # data parmas
        self.input_size = config['input_size']
        self.seq_len = config['seq_len']
        
        # network params
        self.num_cells = config['num_cells']
        self.cells_hidden  = config['cells_hidden']
        self.cells_kernel  = config['cells_kernel']
        self.cells_padding = config['cells_padding']
        self.cells_stride  = config['cells_stride']
        
        if 'dropout_rate' in config:
            if config['dropout_rate'] is not None:
                self.dropout = ComplexDropoutRealFormat(config['dropout_rate'])
            else:
                self.dropout = None
        else:
            self.dropout = None
       

        if len(self.cells_hidden) != self.num_cells:
            raise ValueError("The length of cells_hidden list must be equal as num_cells, error {} != {}".format(len(self.cells_hidden), self.num_cells))
        if len(self.cells_kernel) != self.num_cells:
            raise ValueError("The length of cells_kernel list must be equal as num_cells, error {} != {}".format(len(self.cells_kernel), self.num_cells))
        if len(self.cells_padding) != self.num_cells:
            raise ValueError("The length of cells_padding list must be equal as num_cells, error {} != {}".format(len(self.cells_padding), self.num_cells))
            
        self.build_cell()
        
        # for param_tensor in self.state_dict():
        #     print(param_tensor, "\t", self.state_dict()[param_tensor].size())

            
        
    def build_cell(self):
        self.cells = []
        for h_size, k_size, padding, stride in zip(self.cells_hidden, self.cells_kernel, self.cells_padding, self.cells_stride):
            if self.cells == []:
                cell = ComplexConvGRUCell_RealGate(self.input_size, h_size, k_size, padding=padding, stride=stride, bias=True, dropout=self.dropout, device=self.device)
                self.cells.append(cell)
            else:
                cell = ComplexConvGRUCell_RealGate(last_h_size, h_size, k_size, padding=padding, stride=stride, bias=True, dropout=self.dropout, device=self.device)
                self.cells.append(cell)
            last_h_size = h_size
        self.cells = nn.ModuleList(self.cells)
        # self.out_conv_1x1 = Complex2RealConv(int(last_h_size * self.seq_len), 1, kernel_size=3, padding=1, bias=True, device=self.device)
        self.out_conv_1x1 = nn.Conv2d(int(last_h_size * self.seq_len), 1, kernel_size=1, padding=0, bias=True)# .to(self.device)
        # self.out_conv_1x1 = Complex2RealConv(int(last_h_size * self.seq_len), 1, kernel_size=1, padding=0, bias=True, device=self.device)
        
    def set_zero_state(self, input):
        # set default zero state with shape (batch_size, first_hidden_size)
        zero_state = []
        for h_size in self.hidden_size:
            # zero_state.append(torch.zeros(input.size(0), h_size, dtype=input.dtype, device=input.device).to(self.device))
            zero_state.append(torch.zeros(input.size(0), h_size, dtype=input.dtype, device=input.device))
        return zero_state
        
    def forward(self, input_data, state_data=None):
        # input_data <torch.complex64> with shape (batch, channel, h, w)
        # default state_data is None. 
        hx = [None] * self.num_cells
        hx[0] = state_data  
        
        hidden_output = []      
        
        for x in range(self.seq_len):
            for cell_no, cell in enumerate(self.cells):
                # print(x, cell_no)
                if cell_no == 0:
                    hx[cell_no] = cell(input_data[:, x, :, :], hx[cell_no])
                else:
                    hx[cell_no] = cell(hx[cell_no-1], hx[cell_no])
                
                # print(hx[cell_no][0][0][0][0])
                # print('-*-*-*-*-*')
            hidden_output.append(hx[-1])
        
        # hx[-1] with shape (batch, last_h_size, h, w)
        # concat_output with shape (batch, channel, h, w)
        concat_output = torch.cat(hidden_output, dim=1)
        # concat_output = (torch.mean(concat_output, dim=1)).unsqueeze(1)
        # output = torch.abs(concat_output)
        output = F.relu(self.out_conv_1x1(torch.abs(concat_output)), inplace=True)
        # output = F.relu(self.out_conv_1x1(concat_output), inplace=True)

        return output
        
        
        
    
# ConvGRU_network = ConvGRU(25, 1, 3, 1, 'cpu')
'''
config = {
    'input_size': 1,
    'seq_len': 25,
    'num_cells': 3,
    'cells_hidden': [3, 8, 1],
    'cells_kernel': [3, 3, 3],
    'cells_padding':[1, 1, 1],
    'cells_stride': [1, 1, 1]
}

network = ComplexConvGRU(config, 'cuda')
# network_real = ConvGRU(1, [3, 16, 1], [3, 3, 3], 3)
x = torch.randn(1, 25, 96, 1024, dtype=torch.complex64).to('cuda') 
# x_real = torch.ones(10, 25, 1024, 96)
#output = network(x)
# output_real = network_real(x_real)

#print(output.shape, output.dtype)
print(output, torch.mean(output))

from thop import profile
flops,param = profile(network,inputs=(x,))
print(flops)
'''
        
