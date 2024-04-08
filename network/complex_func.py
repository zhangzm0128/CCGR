import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

EPSILON = 1e-6
POSINF = 2e32
NEGINF = -2e32


def complex_exp(x):
    in_r, in_i = x.real, x.imag
    out_r = torch.exp(in_r) * torch.cos(in_i)
    out_i = torch.exp(in_r) * torch.sin(in_i)
    return out_r + 1j * out_i

def complex_sigmoid(x):
    # Vesion 1
    # https://arxiv.org/abs/1806.08267 (Complex gate recurrent nerual network)
    # sigmoid_alpha = torch.sigmoid(torch.Tensor([0.0]).to(x.device))
    # sigmoid_beta = torch.sigmoid(torch.Tensor([1.0]).to(x.device))
    # in_real, in_imag = x.real, x.imag
    # out_real = in_real * sigmoid_alpha + in_imag * sigmoid_beta
    # return out_real + 1j * torch.zeros_like(in_real)

    # Vesion 2   
    # https://github.com/omrijsharon/torchlex/blob/master/torchlex.py
    in_r, in_i = x.real, x.imag
    d = 1 + 2 * torch.exp(-in_r) * torch.cos(in_i) + torch.exp(-2 * in_r) + EPSILON
    out_r = 1 + torch.exp(-in_r) * torch.cos(in_i) / d
    out_i = torch.exp(-in_r) * torch.sin(in_i) / d
    return out_r + 1j * out_i
    
def complex_split_sigmoid(x):
    return torch.sigmoid(x.real) + 1j * torch.sigmoid(x.imag)

def complex_split_tanh(x):
    return torch.tanh(x.real) + 1j * torch.tanh(x.imag)
    
def complex_tanh(x):
    # Version 1
    # T. Kim and T. Adali, “Fully complex backpropagation for constant envelope signal processing,” 
    # in Proc. Neural Networks for Signal Processing X. IEEE Signal Processing Society Workshop, vol. 1, 2000, pp. 231–240 vol.1.
    
    # return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    return (complex_exp(x) - complex_exp(-x)) / (complex_exp(x) + complex_exp(-x))
    
    # version 2
    # https://github.com/omrijsharon/torchlex/blob/master/torchlex.py
    # in_r, in_i = x.real, x.imag
    # print('complex_tanh')
    # d = torch.cosh(2 * in_r) + torch.cos(2 * in_i) + EPSILON
    # d = torch.nan_to_num(d, posinf=POSINF, neginf=NEGINF)
    # print('d:', torch.any(torch.isnan(d)), torch.sum(torch.isnan(d)))
    # out_r = torch.sinh(2 * in_r) / d
    # print('out_r:', torch.any(torch.isnan(out_r)), torch.sum(torch.isnan(out_r)))
    # if torch.any(torch.isnan(out_r)):
    #     print(in_r[torch.isnan(out_r)])  
    #     print(in_i[torch.isnan(out_r)])
    #     print(d[torch.isnan(out_r)])
    # out_i = torch.sin(2 * in_i) / d
    # print('out_i:', torch.any(torch.isnan(out_i)), torch.sum(torch.isnan(out_i)))
    # return out_r + 1j * out_i
    
    
def real2complex(real_func, imag_func, x, dtype=torch.complex64):
    out_real = (real_func(x.real) - imag_func(x.imag))# .type(dtype)
    out_imag = (real_func(x.imag) + imag_func(x.real))# .type(dtype)
        
    return out_real + 1j * out_imag
    
def complex2real(real_func, imag_func, x, dtype):
    pass
    
def init_complex_conv_weight(kernel_size, input_dim, output_dim, init_mode='HeInit', device='cpu'):
    n_row = output_dim * input_dim
    if type(kernel_size) not in [tuple, list]:
        kernel_size = (kernel_size, kernel_size)
    n_col = kernel_size[0] * kernel_size[1]
    
    r = torch.randn(n_row, n_col).to(device)
    i = torch.randn(n_row, n_col).to(device)
    z = r + 1j * i
    
    u, _, v = torch.linalg.svd(z)
    z_unitary = u @ ((torch.eye(n_row, n_col).to(device) + 1j * torch.zeros(n_row, n_col).to(device)) @ torch.conj(v).T)
    real_unitary = z_unitary.real
    imag_unitary = z_unitary.imag
    
    real_reshape = real_unitary.reshape(n_row, kernel_size[0], kernel_size[1])
    imag_reshape = imag_unitary.reshape(n_row, kernel_size[0], kernel_size[1])
    
    if init_mode == 'HeInit':
        desired_var = 1. / input_dim
    elif init_mode == 'GlorotInit':
        desired_var = 1. / (input_dim + output_dim)
    else:
        raise ValueError("Error init_mode: {}, which is limited in ['HeInit', 'GlorotInit']".format(init_mode))
        
    real_scale = torch.sqrt(desired_var / torch.var(real_reshape)) * real_reshape
    imag_scale = torch.sqrt(desired_var / torch.var(imag_reshape)) * imag_reshape
    
    real_weight = real_scale.reshape(output_dim, input_dim, kernel_size[0], kernel_size[1])
    imag_weight = imag_scale.reshape(output_dim, input_dim, kernel_size[0], kernel_size[1])
    
    return real_weight, imag_weight
    
    
def init_complex_fc(input_dim, output_dim, bias=True, device='cpu'):
    # Glorot initialization
    weight = torch.empty(input_dim * 2, output_dim).to(device)
    weight = nn.init.xavier_uniform_(weight, gain=1.0)
    weight = nn.Parameter(weight, requires_grad=True).to(device)
    if bias == True:
        bias = torch.zeros(output_dim).to(device)
        bias = nn.Parameter(bias, requires_grad=True).to(device)
        return torch.complex(weight[:input_dim, :], weight[input_dim:, :]), bias
    else:
        return weight, None
    
    
def zRelu(x):
    # Paper: Nitzan Guberman. On complex valued convolutional neural networks. arXiv preprint arXiv:1602.09046, 2016
    # Code: https://github.com/omrijsharon/torchlex/blob/5da61b615323f613cd0bb1e780f71c55d3c01250/torchlex.py#L147
    in_real, in_imag = x.real, x.imag
    x_angle = compute_angle(x)
    mask = ((0 < x_angle) * (x_angle < torch.pi / 2))
    out_real = in_real * mask
    out_imag = in_imag * mask
    return out_real + 1j * out_imag
    
def split_Relu(x):
    return F.relu(x.real) + 1j * F.relu(x.imag)

def mod_Relu(x, bias):
    # Paper: Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. arXiv preprint arXiv:1511.06464, 2015.
    # Code: https://github.com/omrijsharon/torchlex/blob/5da61b615323f613cd0bb1e780f71c55d3c01250/torchlex.py#L147
    in_real, in_imag = x.real, x.imag
    in_mag = torch.sqrt(in_real ** 2 + in_imag ** 2)
    mask = ((in_mag + bias) >= 0).float() * (1 + bias / in_mag)
    out_real = mask * in_real
    out_imag = mask * in_imag
    return out_real + 1j * out_imag
    
class modRelu(nn.Module):
    def __init__(self, device='cpu'):
        super(modRelu, self).__init__()
        self.device = device
        self.bias = nn.Parameter(torch.FloatTensor([1]).uniform_(-0.01, 0.01)).to(self.device)
    def forward(self, x):
        return mod_Relu(x, self.bias)
        
    
def compute_angle(x):
    x_real, x_imag = x.real, x.imag
    theta = torch.atan(x_imag / x_real)
    theta[x_real < 0] += torch.pi
    return theta
    

def opposite(x):
    return -x.real + 1j * (-x.imag)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
